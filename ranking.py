import logging
import math
from typing import Dict, List

import pandas as pd

from models import PatientRequest, HospitalWithDistance
from disease_mapping import DiseaseMapping, KTASMapping

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 거리 계산 (Haversine 공식, 미터 단위)"""
    R = 6371000  # 지구 반지름 (미터)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_distance_penalty(distance_km: float) -> float:
    """거리에 따른 점수 패널티 (0~1)"""
    if distance_km <= 10:
        return 1.0      # 10km 이내: 패널티 없음
    elif distance_km <= 30:
        return 0.8      # 30km 이내: 20% 감소
    elif distance_km <= 50:
        return 0.5      # 50km 이내: 50% 감소
    elif distance_km <= 100:
        return 0.3      # 100km 이내: 70% 감소
    else:
        return 0.1      # 100km 초과: 90% 감소


class HospitalRankingEngine:
    """병원 랭킹 계산 엔진 - 위치 기반"""

    # 응급차량 평균 속도 (m/s) - 40km/h 가정
    AVG_SPEED_MPS = 40 * 1000 / 3600  # 약 11.1 m/s

    def __init__(self, data_processor):
        self.data_processor = data_processor

    def estimate_travel_time(self, distance_meters: float) -> float:
        """거리 기반 이동 시간 추정 (초 단위)"""
        return distance_meters / self.AVG_SPEED_MPS

    def find_nearby_hospitals(
        self,
        lat: float,
        lng: float,
        radius_km: float = 100.0
    ) -> pd.DataFrame:
        """Haversine 공식으로 근처 병원 검색"""
        if self.data_processor.hospital_data is None:
            logger.error("병원 데이터가 로드되지 않았습니다.")
            return pd.DataFrame()

        df = self.data_processor.hospital_data.copy()

        # 거리 계산
        df['distance_meters'] = df.apply(
            lambda row: haversine_distance(
                lat, lng,
                row['LATITUDE'], row['LONGITUDE']
            ),
            axis=1
        )
        df['distance_km'] = df['distance_meters'] / 1000.0

        # 반경 내 병원만 필터링
        nearby = df[df['distance_km'] <= radius_km].copy()

        # 이동 시간 추정
        nearby['duration_seconds'] = nearby['distance_meters'].apply(self.estimate_travel_time)
        nearby['duration_minutes'] = nearby['duration_seconds'] / 60.0

        logger.info(f"반경 {radius_km}km 내 병원: {len(nearby)}개 (전체 {len(df)}개)")

        return nearby

    def filter_capable_hospitals(
        self,
        hospitals: pd.DataFrame,
        disease_code: str
    ) -> pd.DataFrame:
        """의료 역량 기준으로 병원 필터링"""
        if hospitals.empty:
            return hospitals

        requirements = DiseaseMapping.get_requirements(disease_code)
        required_codes = requirements['service_codes']
        requires_trauma = requirements.get('requires_trauma_center', False)

        def has_required_capability(row):
            # 서비스 코드로 확인
            has_service = any(
                code in row['service_codes'] for code in required_codes
            )

            # Trauma Center 요구사항 확인
            if requires_trauma:
                has_trauma = row['has_trauma_center'] == 1
                return has_service and has_trauma

            return has_service

        hospitals = hospitals.copy()
        hospitals['is_capable'] = hospitals.apply(has_required_capability, axis=1)
        capable_hospitals = hospitals[hospitals['is_capable']].copy()

        category = DiseaseMapping.get_category(disease_code)
        logger.info(f"질환 '{category}' (코드: {disease_code}): {len(hospitals)}개 중 {len(capable_hospitals)}개 병원이 역량 보유")

        return capable_hospitals

    def calculate_scores(
        self,
        capable_hospitals: pd.DataFrame,
        severity_code: str
    ) -> pd.DataFrame:
        """병원별 점수 계산"""
        if capable_hospitals.empty:
            return capable_hospitals

        df = capable_hospitals.copy()

        # KTAS 정보 가져오기
        ktas_info = KTASMapping.get_level_info(severity_code)
        ktas_level = ktas_info['level']
        time_weight = ktas_info['time_weight']
        hospital_weight = ktas_info['hospital_weight']

        # 1. Time Score (시간 기반 점수)
        # duration_minutes가 작을수록 점수가 높음
        df['time_score'] = (1 / (df['duration_minutes'] + 1)) * 1000000 * time_weight

        # 2. Distance Penalty (거리 패널티)
        df['distance_penalty'] = df['distance_km'].apply(calculate_distance_penalty)
        df['time_score'] = df['time_score'] * df['distance_penalty']

        # 3. Cluster Weight (병원 규모 가중치)
        cluster_weights = {
            0: 1.0,   # 대형 병원
            1: 0.5,   # 소형 병원
            2: 0.75   # 중형 병원
        }

        # 중증도에 따른 병원 규모 가중치 조정
        if ktas_level <= 2:
            # 중증: 대형 병원 강력 선호
            df['cluster_score'] = df['cluster'].map(cluster_weights) * 200000 * hospital_weight
        elif ktas_level >= 4:
            # 경증: 근거리 우선, 병원 규모 덜 중요
            df['cluster_score'] = df['cluster'].map(cluster_weights) * 50000
        else:
            # 중등도
            df['cluster_score'] = df['cluster'].map(cluster_weights) * 100000

        # 4. Resource Score (병상 수 기반)
        max_beds = df['TOTAL_BEDS'].max()
        if max_beds > 0:
            df['resource_score'] = (df['TOTAL_BEDS'] / max_beds) * 200000
        else:
            df['resource_score'] = 0

        # 5. 최종 점수 계산
        # 중증도에 따른 가중치 조정
        if ktas_level <= 2:
            # 중증: 시간 50%, 병원규모 30%, 자원 20%
            df['final_score'] = (
                df['time_score'] * 0.5 +
                df['cluster_score'] * 0.3 +
                df['resource_score'] * 0.2
            )
        elif ktas_level >= 4:
            # 경증: 시간 70%, 병원규모 15%, 자원 15%
            df['final_score'] = (
                df['time_score'] * 0.7 +
                df['cluster_score'] * 0.15 +
                df['resource_score'] * 0.15
            )
        else:
            # 중등도: 기본 배분
            df['final_score'] = (
                df['time_score'] * 0.6 +
                df['cluster_score'] * 0.2 +
                df['resource_score'] * 0.2
            )

        # 6. 추천 사유 생성
        df['recommendation_reason'] = df.apply(
            lambda row: self._generate_reason(row, severity_code),
            axis=1
        )

        return df

    def _generate_reason(self, row, severity_code: str) -> str:
        """추천 사유 생성"""
        reasons = []

        duration_min = row['duration_minutes']
        distance_km = row['distance_km']

        # 거리/시간 기반
        if duration_min < 10:
            reasons.append(f"최단거리 ({duration_min:.1f}분, {distance_km:.1f}km)")
        elif duration_min < 20:
            reasons.append(f"근거리 ({duration_min:.1f}분, {distance_km:.1f}km)")
        else:
            reasons.append(f"소요시간 {duration_min:.1f}분 ({distance_km:.1f}km)")

        # 병원 규모/역량
        cluster_names = {0: '대형거점병원', 1: '지역병원', 2: '중형병원'}
        reasons.append(cluster_names.get(row['cluster'], ''))

        # 특수 센터
        if row['has_trauma_center'] == 1:
            reasons.append('외상센터 보유')

        # 병상 수
        if row['TOTAL_BEDS'] >= 300:
            reasons.append(f'대규모 병상({int(row["TOTAL_BEDS"])}병상)')

        return ' | '.join([r for r in reasons if r])

    def rank_hospitals_by_location(
        self,
        patient: PatientRequest,
        top_n: int = 10
    ) -> List[Dict]:
        """전체 랭킹 파이프라인 - 위치 기반"""

        logger.info(f"환자 위치: ({patient.latitude}, {patient.longitude})")
        logger.info(f"질병 코드: {patient.disease_code}, 중증도: {patient.severity_code}")

        # Step 1: 근처 병원 검색 (100km 반경)
        nearby_hospitals = self.find_nearby_hospitals(
            patient.latitude,
            patient.longitude,
            radius_km=100.0
        )

        if nearby_hospitals.empty:
            logger.warning("반경 100km 내 병원이 없습니다.")
            return []

        # Step 2: 의료 역량 필터링
        capable_hospitals = self.filter_capable_hospitals(
            nearby_hospitals,
            patient.disease_code
        )

        if capable_hospitals.empty:
            category = DiseaseMapping.get_category(patient.disease_code)
            logger.warning(f"질환 '{category}'에 적합한 병원이 없습니다.")
            return []

        # Step 3: 점수 계산
        scored_hospitals = self.calculate_scores(
            capable_hospitals,
            patient.severity_code
        )

        # Step 4: 정렬 및 상위 N개 선택
        ranked = scored_hospitals.sort_values('final_score', ascending=False).head(top_n)

        # Step 5: 결과 포맷팅
        results = []
        for idx, (_, row) in enumerate(ranked.iterrows(), 1):
            results.append({
                'rank': idx,
                'facid': row['FACID'],
                'name': row['FACNAME'],
                'address': f"{row['ADDRESS']}, {row['CITY']}",
                'county': row['COUNTY_NAME'],
                'latitude': float(row['LATITUDE']),
                'longitude': float(row['LONGITUDE']),
                'distance_meters': int(row['distance_meters']),
                'distance_km': round(row['distance_km'], 2),
                'duration_seconds': int(row['duration_seconds']),
                'duration_minutes': round(row['duration_minutes'], 1),
                'total_beds': int(row['TOTAL_BEDS']),
                'cluster': int(row['cluster']),
                'cluster_name': {0: '대형/거점', 1: '소형', 2: '중형'}[row['cluster']],
                'has_trauma_center': bool(row['has_trauma_center']),
                'final_score': round(row['final_score'], 2),
                'time_score': round(row['time_score'], 2),
                'cluster_score': round(row['cluster_score'], 2),
                'resource_score': round(row['resource_score'], 2),
                'distance_penalty': round(row['distance_penalty'], 2),
                'recommendation_reason': row['recommendation_reason']
            })

        logger.info(f"Top {len(results)} 병원 랭킹 생성 완료")
        return results

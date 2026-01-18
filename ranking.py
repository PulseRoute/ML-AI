import logging
from typing import Dict, List

import pandas as pd

from models import HospitalCandidate
from ems_mapping import EMSCodeMapping

logger = logging.getLogger(__name__)


class HospitalRankingEngine:
    """병원 랭킹 계산 엔진"""
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
    
    def filter_capable_hospitals(
        self,
        candidates: List[HospitalCandidate],
        ems_code: str
    ) -> pd.DataFrame:
        """의료 역량 기준으로 병원 필터링"""
        
        requirements = EMSCodeMapping.get_requirements(ems_code)
        required_codes = requirements['service_codes']
        required_names = requirements['service_names']
        requires_trauma = requirements.get('requires_trauma_center', False)
        
        # 후보 병원 데이터 가져오기
        candidate_facids = [c.facid for c in candidates]
        hospital_subset = self.data_processor.hospital_data[
            self.data_processor.hospital_data['FACID'].isin(candidate_facids)
        ].copy()
        
        # 백엔드 데이터 매핑
        duration_map = {c.facid: c.duration for c in candidates}
        distance_map = {c.facid: c.distance for c in candidates}
        
        hospital_subset['duration'] = hospital_subset['FACID'].map(duration_map)
        hospital_subset['distance'] = hospital_subset['FACID'].map(distance_map)
        
        # 의료 역량 필터링
        def has_required_capability(row):
            # 서비스 코드로만 확인 (숫자 코드 통일)
            has_service = any(
                code in row['service_codes'] for code in required_codes
            )
            
            # Trauma Center 요구사항 확인
            if requires_trauma:
                has_trauma = row['has_trauma_center'] == 1
                return has_service and has_trauma
            
            return has_service
        
        hospital_subset['is_capable'] = hospital_subset.apply(has_required_capability, axis=1)
        capable_hospitals = hospital_subset[hospital_subset['is_capable']].copy()
        
        logger.info(f"EMS 코드 '{ems_code}': {len(hospital_subset)}개 중 {len(capable_hospitals)}개 병원이 역량 보유")
        
        return capable_hospitals
    
    def calculate_scores(
        self,
        capable_hospitals: pd.DataFrame,
        triage_priority: int
    ) -> pd.DataFrame:
        """병원별 점수 계산"""
        
        if capable_hospitals.empty:
            return capable_hospitals
        
        df = capable_hospitals.copy()
        
        # 1. Time Score (60%) - 가장 중요
        # duration은 초 단위, 분으로 변환하여 계산
        df['duration_minutes'] = df['duration'] / 60.0
        df['time_score'] = (1 / (df['duration_minutes'] + 1)) * 1000000
        
        # 2. Clustering Weight (20%) - 중증도에 따른 가중치
        cluster_weights = {
            0: 1.0,   # 대형 병원
            1: 0.5,   # 소형 병원
            2: 0.75   # 중형 병원
        }
        
        # 중증 환자(1-2)는 대형 병원 선호, 경증(4-5)은 덜 중요
        if triage_priority <= 2:
            # 중증: 클러스터 가중치 2배 증폭
            df['cluster_score'] = df['cluster'].map(cluster_weights) * 200000
        elif triage_priority >= 4:
            # 경증: 클러스터 가중치 감소
            df['cluster_score'] = df['cluster'].map(cluster_weights) * 50000
        else:
            # 중등도
            df['cluster_score'] = df['cluster'].map(cluster_weights) * 100000
        
        # 3. Resource Score (20%) - 병상 수 기반
        max_beds = df['TOTAL_BEDS'].max()
        if max_beds > 0:
            df['resource_score'] = (df['TOTAL_BEDS'] / max_beds) * 200000
        else:
            df['resource_score'] = 0
        
        # 4. 최종 점수 계산
        df['final_score'] = (
            df['time_score'] * 0.6 +
            df['cluster_score'] * 0.2 +
            df['resource_score'] * 0.2
        )
        
        # 5. 추천 사유 생성
        df['recommendation_reason'] = df.apply(self._generate_reason, axis=1)
        
        return df
    
    def _generate_reason(self, row) -> str:
        """추천 사유 생성"""
        reasons = []
        
        duration_min = row['duration_minutes']
        
        # 시간 기반
        if duration_min < 10:
            reasons.append(f"최단거리 ({duration_min:.1f}분)")
        elif duration_min < 20:
            reasons.append(f"근거리 ({duration_min:.1f}분)")
        else:
            reasons.append(f"소요시간 {duration_min:.1f}분")
        
        # 병원 규모/역량
        cluster_names = {0: '대형거점병원', 1: '지역병원', 2: '중형병원'}
        reasons.append(cluster_names.get(row['cluster'], ''))
        
        # 특수 센터
        if row['has_trauma_center'] == 1:
            reasons.append('외상센터 보유')
        
        # 병상 수
        if row['TOTAL_BEDS'] >= 300:
            reasons.append(f'대규모 병상({int(row["TOTAL_BEDS"]) }병상)')
        
        return ' | '.join(reasons)
    
    def rank_hospitals(
        self,
        candidates: List[HospitalCandidate],
        ems_code: str,
        triage_priority: int,
        top_n: int = 10
    ) -> List[Dict]:
        """병원 랭킹 생성 - 메인 함수"""
        
        # Step 1: 의료 역량 필터링
        capable_hospitals = self.filter_capable_hospitals(candidates, ems_code)
        
        if capable_hospitals.empty:
            logger.warning(f"⚠️ EMS 코드 '{ems_code}'에 적합한 병원이 없습니다.")
            return []
        
        # Step 2: 점수 계산
        scored_hospitals = self.calculate_scores(capable_hospitals, triage_priority)
        
        # Step 3: 정렬 및 상위 N개 선택
        ranked = scored_hospitals.sort_values('final_score', ascending=False).head(top_n)
        
        # Step 4: 결과 포맷팅
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
                'duration_seconds': int(row['duration']),
                'duration_minutes': round(row['duration_minutes'], 1),
                'distance_meters': int(row['distance']),
                'total_beds': int(row['TOTAL_BEDS']),
                'cluster': int(row['cluster']),
                'cluster_name': {0: '대형/거점', 1: '소형', 2: '중형'}[row['cluster']],
                'has_trauma_center': bool(row['has_trauma_center']),
                'final_score': round(row['final_score'], 2),
                'time_score': round(row['time_score'], 2),
                'cluster_score': round(row['cluster_score'], 2),
                'resource_score': round(row['resource_score'], 2),
                'recommendation_reason': row['recommendation_reason']
            })
        
        logger.info(f"✅ Top {len(results)} 병원 랭킹 생성 완료")
        return results

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HospitalDataProcessor:
    """병원 데이터 전처리 및 클러스터링 - 싱글톤 패턴"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HospitalDataProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.hospital_data = None
        self.kmeans_model = None
        self.scaler = None
        self._initialized = True
        
    def load_and_process_data(self, data_dir: str):
        """데이터 로드 및 전처리"""
        logger.info("병원 데이터 로딩 시작...")
        
        try:
            # CSV 파일 로드
            location_df = pd.read_csv(
                os.path.join(data_dir, 'location.csv'),
                encoding='utf-8-sig',
                dtype={'FACID': str}
            )
            services_df = pd.read_csv(
                os.path.join(data_dir, 'services.csv'),
                encoding='utf-8-sig',
                dtype={'FACID': str}
            )
            ed_stats_df = pd.read_csv(
                os.path.join(data_dir, 'ed_stats.csv'),
                dtype={'FACID': str}
            )
            
            logger.info(f"Location: {len(location_df)} rows, Services: {len(services_df)} rows, ED Stats: {len(ed_stats_df)} rows")
            
            # GACH (General Acute Care Hospital)만 필터링
            location_df = location_df[location_df['FAC_TYPE_CODE'] == 'GACH'].copy()
            logger.info(f"GACH 병원 필터링 후: {len(location_df)} rows")
            
            # 기본 컬럼 선택 및 정제
            location_df = location_df[[
                'FACID', 'FACNAME', 'ADDRESS', 'CITY', 'COUNTY_NAME',
                'LATITUDE', 'LONGITUDE', 'TRAUMA_CTR', 'TRAUMA_PED_CTR',
                'CRITICAL_ACCESS_HOSPITAL', 'FAC_TYPE_CODE'
            ]].copy()
            
            # NaN 처리
            location_df['LATITUDE'] = pd.to_numeric(location_df['LATITUDE'], errors='coerce')
            location_df['LONGITUDE'] = pd.to_numeric(location_df['LONGITUDE'], errors='coerce')
            location_df = location_df.dropna(subset=['LATITUDE', 'LONGITUDE'])
            
            # 병상 수 계산 - EMS 관련 급성기 병상만 집계
            # 응급 환자 수용 가능한 병상 타입만 필터링
            ems_relevant_bed_types = [
                'GENERAL ACUTE CARE HOSPITAL',
                'UNSPECIFIED GENERAL ACUTE CARE',
                'INTENSIVE CARE',
                'PEDIATRIC INTENSIVE CARE UNIT',
                'CORONARY CARE',
                'BURN',
                'ACUTE RESPIRATORY CARE',
                'PERINATAL',
                'PEDIATRIC',
                'INTERMEDIATE CARE',
                'REHABILITATION'  # 급성기 재활도 포함
            ]
            
            ed_stats_filtered = ed_stats_df[
                ed_stats_df['BED_CAPACITY_TYPE'].isin(ems_relevant_bed_types)
            ].copy()
            
            bed_capacity = ed_stats_filtered.groupby('FACID')['BED_CAPACITY'].sum().reset_index()
            bed_capacity.columns = ['FACID', 'TOTAL_BEDS']
            
            logger.info(f"병상 집계: 전체 {len(ed_stats_df)} rows → EMS 관련 {len(ed_stats_filtered)} rows")
            
            # 서비스 리스트 생성
            services_list = services_df.groupby('FACID').agg({
                'SERVICE_TYPE_CODE': lambda x: list(x.unique()),
                'SERVICE_TYPE_NAME': lambda x: list(x.unique())
            }).reset_index()
            services_list.columns = ['FACID', 'service_codes', 'service_names']
            
            # 데이터 병합
            self.hospital_data = location_df.merge(bed_capacity, on='FACID', how='left')
            self.hospital_data = self.hospital_data.merge(services_list, on='FACID', how='left')
            
            # 병상 수 결측치 처리
            self.hospital_data['TOTAL_BEDS'] = self.hospital_data['TOTAL_BEDS'].fillna(0)
            
            # 서비스 리스트 결측치 처리
            self.hospital_data['service_codes'] = self.hospital_data['service_codes'].apply(
                lambda x: x if isinstance(x, list) else []
            )
            self.hospital_data['service_names'] = self.hospital_data['service_names'].apply(
                lambda x: x if isinstance(x, list) else []
            )
            
            # Feature Engineering
            self.hospital_data['log_beds'] = np.log1p(self.hospital_data['TOTAL_BEDS'])
            self.hospital_data['has_trauma_center'] = (
                self.hospital_data['TRAUMA_CTR'].notna() & 
                (self.hospital_data['TRAUMA_CTR'] != '')
            ).astype(int)
            self.hospital_data['service_count'] = self.hospital_data['service_codes'].apply(len)
            
            # Clustering 수행
            self._perform_clustering()
            
            logger.info(f"✅ 데이터 로딩 완료: {len(self.hospital_data)} 병원")
            logger.info(f"클러스터 분포:\n{self.hospital_data['cluster'].value_counts()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 데이터 로딩 실패: {str(e)}")
            raise
    
    def _perform_clustering(self):
        """KMeans 클러스터링 수행"""
        features_for_clustering = self.hospital_data[[
            'log_beds', 'service_count', 'has_trauma_center'
        ]].copy()
        
        # 스케일링
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_for_clustering)
        
        # KMeans 클러스터링
        self.kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.hospital_data['cluster'] = self.kmeans_model.fit_predict(features_scaled)
        
        # 클러스터 해석 (평균 병상 수 기준)
        cluster_means = self.hospital_data.groupby('cluster')['TOTAL_BEDS'].mean().sort_values(ascending=False)
        cluster_mapping = {
            cluster_means.index[0]: 0,  # 대형 병원
            cluster_means.index[1]: 2,  # 중형 병원
            cluster_means.index[2]: 1   # 소형 병원
        }
        self.hospital_data['cluster'] = self.hospital_data['cluster'].map(cluster_mapping)
        
        logger.info("클러스터별 평균 병상 수:")
        for cluster in [0, 1, 2]:
            avg_beds = self.hospital_data[self.hospital_data['cluster'] == cluster]['TOTAL_BEDS'].mean()
            cluster_name = {0: '대형/거점', 1: '소형', 2: '중형'}[cluster]
            logger.info(f"  Cluster {cluster} ({cluster_name}): {avg_beds:.1f} beds")
    
    def get_hospital_info(self, facid: str) -> Optional[Dict]:
        """특정 병원 정보 조회"""
        hospital = self.hospital_data[self.hospital_data['FACID'] == facid]
        if hospital.empty:
            return None
        return hospital.iloc[0].to_dict()

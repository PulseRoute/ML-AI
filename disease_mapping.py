from typing import Dict, Optional


class DiseaseMapping:
    """ICD-10 코드 기반 질환-의료서비스 매핑"""

    # ICD-10 주요 응급 질환 그룹
    # 서비스 코드 참조:
    #   2 = BASIC EMERGENCY MEDICAL
    #   4 = CARDIAC CATHETERIZATION LABORATORY SERVICES
    #   8 = COMPREHENSIVE EMERGENCY MEDICAL SERVICES
    #   23 = RESPIRATORY CARE SERVICES
    #   227 = INTENSIVE CARE SERVICE
    ICD10_MAPPINGS = {
        # 심근경색 (I21.x)
        'I21': {
            'category': 'STEMI/NSTEMI',
            'service_codes': [4, 2, 8],
            'service_names': ['CARDIAC CATHETERIZATION', 'EMERGENCY MEDICAL', 'COMPREHENSIVE EMERGENCY'],
            'requires_trauma_center': False,
            'description': '급성 심근경색'
        },
        'I22': {
            'category': 'STEMI/NSTEMI',
            'service_codes': [4, 2, 8],
            'service_names': ['CARDIAC CATHETERIZATION', 'EMERGENCY MEDICAL', 'COMPREHENSIVE EMERGENCY'],
            'requires_trauma_center': False,
            'description': '후속 심근경색'
        },
        # 뇌졸중 (I60-I69)
        'I60': {
            'category': 'Stroke',
            'service_codes': [8, 2],
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '비외상성 지주막하 출혈'
        },
        'I61': {
            'category': 'Stroke',
            'service_codes': [8, 2],
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '비외상성 뇌내출혈'
        },
        'I62': {
            'category': 'Stroke',
            'service_codes': [8, 2],
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '기타 비외상성 두개내 출혈'
        },
        'I63': {
            'category': 'Stroke',
            'service_codes': [8, 2],
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '뇌경색'
        },
        'I64': {
            'category': 'Stroke',
            'service_codes': [8, 2],
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '출혈 또는 경색으로 명시되지 않은 뇌졸중'
        },
        'I65': {
            'category': 'Stroke',
            'service_codes': [8, 2],
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '뇌경색을 유발하지 않은 뇌전동맥 폐색/협착'
        },
        'I66': {
            'category': 'Stroke',
            'service_codes': [8, 2],
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '뇌경색을 유발하지 않은 뇌동맥 폐색/협착'
        },
        # 패혈증 (A41)
        'A41': {
            'category': 'Sepsis',
            'service_codes': [8, 2, 227],
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL', 'INTENSIVE CARE'],
            'requires_trauma_center': False,
            'description': '기타 패혈증'
        },
        'A40': {
            'category': 'Sepsis',
            'service_codes': [8, 2, 227],
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL', 'INTENSIVE CARE'],
            'requires_trauma_center': False,
            'description': '연쇄구균 패혈증'
        },
        # 호흡기 (J96, J80 등)
        'J96': {
            'category': 'Respiratory',
            'service_codes': [23, 8, 2],
            'service_names': ['RESPIRATORY CARE', 'COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '호흡부전'
        },
        'J80': {
            'category': 'Respiratory',
            'service_codes': [23, 8, 2],
            'service_names': ['RESPIRATORY CARE', 'COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '급성 호흡곤란 증후군'
        },
        'J18': {
            'category': 'Respiratory',
            'service_codes': [23, 8, 2],
            'service_names': ['RESPIRATORY CARE', 'COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '폐렴'
        },
        # 심장 관련
        'I46': {
            'category': 'Cardiac',
            'service_codes': [4, 8, 2],
            'service_names': ['CARDIAC CATHETERIZATION', 'COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '심정지'
        },
        'I50': {
            'category': 'Cardiac',
            'service_codes': [4, 8, 2],
            'service_names': ['CARDIAC CATHETERIZATION', 'COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '심부전'
        },
        # 기타 응급
        'R57': {
            'category': 'Shock',
            'service_codes': [8, 2],
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'requires_trauma_center': False,
            'description': '쇼크'
        },
    }

    # 외상 코드 범위 (S00-T88)
    TRAUMA_PREFIXES = ['S', 'T']

    # 기본 응급 매핑
    DEFAULT_MAPPING = {
        'category': 'General',
        'service_codes': [8, 2],
        'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
        'requires_trauma_center': False,
        'description': '일반 응급'
    }

    # 외상 매핑
    TRAUMA_MAPPING = {
        'category': 'Trauma',
        'service_codes': [8],
        'service_names': ['COMPREHENSIVE EMERGENCY', 'TRAUMA'],
        'requires_trauma_center': True,
        'description': '중증 외상'
    }

    @classmethod
    def get_requirements(cls, disease_code: str) -> Dict:
        """ICD-10 코드에 해당하는 의료 서비스 요구사항 반환"""
        if not disease_code:
            return cls.DEFAULT_MAPPING

        # 코드 정규화 (대문자, 공백 제거)
        code = disease_code.strip().upper()

        # 외상 코드 체크 (S00-T88)
        if code and code[0] in cls.TRAUMA_PREFIXES:
            return cls.TRAUMA_MAPPING

        # 정확한 매칭 (소수점 포함 코드: I21.3)
        if code in cls.ICD10_MAPPINGS:
            return cls.ICD10_MAPPINGS[code]

        # 상위 코드 매칭 (I21.3 → I21)
        base_code = code.split('.')[0]
        if base_code in cls.ICD10_MAPPINGS:
            return cls.ICD10_MAPPINGS[base_code]

        # 3자리 코드로 검색 (I213 → I21)
        if len(base_code) >= 3:
            prefix = base_code[:3]
            if prefix in cls.ICD10_MAPPINGS:
                return cls.ICD10_MAPPINGS[prefix]

        return cls.DEFAULT_MAPPING

    @classmethod
    def get_category(cls, disease_code: str) -> str:
        """ICD-10 코드의 질환 카테고리 반환"""
        return cls.get_requirements(disease_code)['category']

    @classmethod
    def is_trauma(cls, disease_code: str) -> bool:
        """외상 코드 여부 확인"""
        if not disease_code:
            return False
        code = disease_code.strip().upper()
        return code and code[0] in cls.TRAUMA_PREFIXES


# KTAS 중증도 매핑
class KTASMapping:
    """KTAS 중증도 코드 매핑"""

    LEVELS = {
        'KTAS_1': {
            'level': 1,
            'name': '소생',
            'description': '즉각적인 소생술 필요',
            'time_weight': 1.5,      # 시간 가중치 최대
            'hospital_weight': 2.0,   # 대형병원 최우선
            'distance_tolerance': 0.5  # 거리 허용도 낮음 (시간 우선)
        },
        'KTAS_2': {
            'level': 2,
            'name': '긴급',
            'description': '긴급 치료 필요',
            'time_weight': 1.3,
            'hospital_weight': 1.5,
            'distance_tolerance': 0.7
        },
        'KTAS_3': {
            'level': 3,
            'name': '응급',
            'description': '응급 치료 필요',
            'time_weight': 1.0,
            'hospital_weight': 1.0,
            'distance_tolerance': 1.0
        },
        'KTAS_4': {
            'level': 4,
            'name': '준응급',
            'description': '준응급 치료',
            'time_weight': 0.8,
            'hospital_weight': 0.7,
            'distance_tolerance': 1.3
        },
        'KTAS_5': {
            'level': 5,
            'name': '비응급',
            'description': '비응급 치료',
            'time_weight': 0.5,
            'hospital_weight': 0.5,
            'distance_tolerance': 1.5  # 거리 허용도 높음 (근거리 우선)
        }
    }

    @classmethod
    def get_level_info(cls, severity_code: str) -> Dict:
        """KTAS 코드에 해당하는 중증도 정보 반환"""
        return cls.LEVELS.get(severity_code, cls.LEVELS['KTAS_3'])

    @classmethod
    def get_level_number(cls, severity_code: str) -> int:
        """KTAS 코드의 숫자 레벨 반환 (1-5)"""
        info = cls.get_level_info(severity_code)
        return info['level']

    @classmethod
    def get_time_weight(cls, severity_code: str) -> float:
        """시간 가중치 반환"""
        return cls.get_level_info(severity_code)['time_weight']

    @classmethod
    def get_hospital_weight(cls, severity_code: str) -> float:
        """병원 규모 가중치 반환"""
        return cls.get_level_info(severity_code)['hospital_weight']

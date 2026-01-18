from typing import Dict


class EMSCodeMapping:
    """EMS 증상 코드와 필요 의료 서비스 매핑"""
    
    MAPPINGS = {
        'STEMI': {
            'service_codes': ['004', '005', '008'],  # Cardiac Cath, Cardiovascular Surgery, Comprehensive ER
            'service_names': ['CARDIAC CATHETERIZATION', 'CARDIOVASCULAR SURGERY', 'COMPREHENSIVE EMERGENCY'],
            'description': '심근경색 (STEMI)'
        },
        'NSTEMI': {
            'service_codes': ['004', '002', '008'],  # Cardiac Cath, Basic ER, Comprehensive ER
            'service_names': ['CARDIAC CATHETERIZATION', 'EMERGENCY MEDICAL'],
            'description': '비ST분절상승 심근경색'
        },
        'Stroke': {
            'service_codes': ['008'],  # Comprehensive Emergency
            'service_names': ['COMPREHENSIVE EMERGENCY', 'EMERGENCY MEDICAL'],
            'description': '뇌졸중'
        },
        'Trauma': {
            'service_codes': ['008'],  # Comprehensive Emergency + Trauma Center flag
            'service_names': ['COMPREHENSIVE EMERGENCY', 'TRAUMA'],
            'requires_trauma_center': True,
            'description': '중증 외상'
        },
        'Sepsis': {
            'service_codes': ['008', '002'],  # Comprehensive or Basic ER
            'service_names': ['EMERGENCY MEDICAL', 'INTENSIVE CARE'],
            'description': '패혈증'
        },
        'Respiratory': {
            'service_codes': ['008', '002', '023'],  # ER + Respiratory Care
            'service_names': ['EMERGENCY MEDICAL', 'RESPIRATORY CARE'],
            'description': '호흡기 응급'
        },
        'General': {
            'service_codes': ['008', '002'],  # Any Emergency
            'service_names': ['EMERGENCY MEDICAL'],
            'description': '일반 응급'
        }
    }
    
    @classmethod
    def get_requirements(cls, ems_code: str) -> Dict:
        """EMS 코드에 해당하는 의료 서비스 요구사항 반환"""
        return cls.MAPPINGS.get(ems_code, cls.MAPPINGS['General'])

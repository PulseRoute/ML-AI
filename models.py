from dataclasses import dataclass
from typing import Optional


@dataclass
class HospitalCandidate:
    """백엔드로부터 받는 병원 후보 데이터 구조 (레거시)"""
    facid: str
    duration: float  # 초 단위 소요 시간
    distance: float  # 미터 단위 거리


@dataclass
class PatientRequest:
    """환자 정보 요청 데이터 구조 (새 API)"""
    name: str
    age: int
    gender: str  # M/F
    disease_code: str  # ICD-10 코드
    severity_code: str  # KTAS_1 ~ KTAS_5
    latitude: float
    longitude: float

    @classmethod
    def from_dict(cls, data: dict) -> 'PatientRequest':
        """딕셔너리에서 PatientRequest 객체 생성"""
        location = data.get('location', {})
        return cls(
            name=data.get('name', ''),
            age=int(data.get('age', 0)),
            gender=data.get('gender', 'M'),
            disease_code=data.get('disease_code', ''),
            severity_code=data.get('severity_code', 'KTAS_3'),
            latitude=float(location.get('latitude', 0)),
            longitude=float(location.get('longitude', 0))
        )


@dataclass
class HospitalWithDistance:
    """거리 정보가 포함된 병원 데이터"""
    facid: str
    distance_meters: float
    duration_seconds: float  # 추정 이동 시간

    @property
    def distance_km(self) -> float:
        """거리를 km 단위로 반환"""
        return self.distance_meters / 1000.0

    @property
    def duration_minutes(self) -> float:
        """이동 시간을 분 단위로 반환"""
        return self.duration_seconds / 60.0

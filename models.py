from dataclasses import dataclass


@dataclass
class HospitalCandidate:
    """백엔드로부터 받는 병원 후보 데이터 구조"""
    facid: str
    duration: float  # 초 단위 소요 시간
    distance: float  # 미터 단위 거리

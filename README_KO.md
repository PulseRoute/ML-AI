# 병원 추천 ML 서버

## 프로젝트 소개

이 프로젝트는 응급환자를 위한 병원 추천 시스템의 ML 서버입니다.  
환자의 증상, 중증도, 후보 병원 정보 등을 입력받아, 최적의 병원을 추천해주는 RESTful API를 제공합니다.

## 주요 기능

- 병원 상세 정보 조회
- GACH(일반 급성기 병원) 목록 제공
- EMS 코드(응급 증상 분류) 목록 제공
- 환자 상태 및 후보 병원 리스트 기반 병원 랭킹 예측

## 폴더 구조

```
app.py                # 메인 서버 실행 파일
data_processor.py     # 데이터 전처리 및 병원 정보 관리
ems_mapping.py        # EMS 코드와 서비스 매핑
models.py             # 데이터 모델 정의
ranking.py            # 병원 랭킹 알고리즘
requirements.txt      # 의존성 목록
dataset/              # 병원/서비스/통계 데이터 CSV
```

## 실행 방법

1. 의존성 설치  
   ```
   pip install -r requirements.txt
   ```

2. 서버 실행  
   ```
   python app.py
   ```

3. Swagger 문서:  
   [http://localhost:5000/swagger/](http://localhost:5000/swagger/)

## 사용 예시

- 병원 상세 정보:  
  `GET /api/hospitals/info/<facid>`

- 병원 랭킹 예측:  
  `POST /api/predict/rank`  
  (JSON body: ems_code, triage_priority, hospital_candidates)


from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Optional
import os
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from models import PatientRequest
from disease_mapping import DiseaseMapping, KTASMapping
from data_processor import HospitalDataProcessor
from ranking import HospitalRankingEngine


app = Flask(__name__)
CORS(app)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/swagger/"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "EMS Hospital Recommendation ML API",
        "description": "응급의료서비스(EMS) 전용 병원 추천 시스템 ML 서버 API (ICD-10 기반)",
        "version": "2.0.0",
        "contact": {
            "name": "EMS ML Team",
            "email": "ems-ml@example.com"
        }
    },
    "host": "localhost:25875",
    "basePath": "/",
    "schemes": ["http", "https"],
    "tags": [
        {
            "name": "Health",
            "description": "서버 상태 확인"
        },
        {
            "name": "Hospital Info",
            "description": "병원 정보 조회"
        },
        {
            "name": "Ranking",
            "description": "병원 랭킹 예측"
        },
        {
            "name": "Reference",
            "description": "참조 데이터"
        }
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

data_processor = HospitalDataProcessor()
ranking_engine = None


def _to_native(obj):
    """Recursively convert pandas/numpy types to native Python types for JSON serialization."""
    if obj is None:
        return None

    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}

    try:
        import pandas as _pd
        if isinstance(obj, _pd.Series):
            return _to_native(obj.to_list())
    except Exception:
        pass

    if isinstance(obj, (list, tuple, set)) or (hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes))):
        try:
            return [_to_native(x) for x in obj]
        except Exception:
            pass

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            pass

    return obj


@app.route('/health', methods=['GET'])
def health_check():
    """서버 헬스 체크
    ---
    tags:
      - Health
    responses:
      200:
        description: 서버 정상 동작 중
        schema:
          type: object
          properties:
            status:
              type: string
            service:
              type: string
            version:
              type: string
            hospitals_loaded:
              type: integer
    """
    return jsonify({
        'status': 'healthy',
        'service': 'EMS ML Server',
        'version': '2.0.0',
        'hospitals_loaded': len(data_processor.hospital_data) if data_processor.hospital_data is not None else 0
    })


@app.route('/api/hospitals/info/<facid>', methods=['GET'])
def get_hospital_info(facid):
    """특정 병원 상세 정보 조회
    ---
    tags:
      - Hospital Info
    parameters:
      - name: facid
        in: path
        type: string
        required: true
        description: 병원 시설 ID
    responses:
      200:
        description: 병원 정보 조회 성공
      404:
        description: 병원을 찾을 수 없음
    """
    info = data_processor.get_hospital_info(facid)
    if info is None:
        return jsonify({'error': 'Hospital not found'}), 404

    result = _to_native(info)
    return jsonify(result)


@app.route('/api/hospitals/gach', methods=['GET'])
def get_gach_hospitals():
    """GACH 지원 병원 목록 조회
    ---
    tags:
      - Hospital Info
    responses:
      200:
        description: GACH 병원 리스트
    """
    if data_processor.hospital_data is None:
        return jsonify({'error': 'Hospital data not loaded'}), 500

    df = data_processor.hospital_data.copy()
    results = []

    for _, row in df.iterrows():
        entry = {
            'facid': _to_native(row.get('FACID')),
            'name': _to_native(row.get('FACNAME')),
            'address': _to_native(row.get('ADDRESS')),
            'city': _to_native(row.get('CITY')),
            'county': _to_native(row.get('COUNTY_NAME')),
            'latitude': _to_native(row.get('LATITUDE')),
            'longitude': _to_native(row.get('LONGITUDE')),
            'total_beds': _to_native(row.get('TOTAL_BEDS')) or 0,
            'service_codes': _to_native(row.get('service_codes')) or [],
            'service_names': _to_native(row.get('service_names')) or [],
            'has_trauma_center': _to_native(row.get('has_trauma_center')) or False,
            'cluster': _to_native(row.get('cluster'))
        }
        results.append(entry)

    return jsonify({'count': len(results), 'hospitals': results})


@app.route('/api/predict/rank', methods=['POST'])
def predict_rank():
    """병원 랭킹 예측
    ---
    tags:
      - Ranking
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - age
            - gender
            - disease_code
            - severity_code
            - location
          properties:
            name:
              type: string
              description: Patient name (optional)
            age:
              type: integer
              description: Patient age
            gender:
              type: string
              enum: [M, F]
              description: Patient gender
            disease_code:
              type: string
              description: "ICD-10 disease code (e.g. I21.3, S72.0)"
            severity_code:
              type: string
              enum: [KTAS_1, KTAS_2, KTAS_3, KTAS_4, KTAS_5]
              description: KTAS severity code
            location:
              type: object
              properties:
                latitude:
                  type: number
                  description: Patient latitude
                longitude:
                  type: number
                  description: Patient longitude
    responses:
      200:
        description: Hospital ranking generated successfully
      400:
        description: Bad request
      500:
        description: Server error
    """
    try:
        data = request.get_json()

        # 필수 파라미터 검증
        required_fields = ['age', 'gender', 'disease_code', 'severity_code', 'location']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        location = data.get('location', {})
        if 'latitude' not in location or 'longitude' not in location:
            return jsonify({'error': 'Missing latitude or longitude in location'}), 400

        # PatientRequest 객체 생성
        patient = PatientRequest.from_dict(data)

        # KTAS 코드 검증
        if patient.severity_code not in KTASMapping.LEVELS:
            return jsonify({'error': f'Invalid severity_code. Must be one of: {list(KTASMapping.LEVELS.keys())}'}), 400

        logger.info(f"랭킹 요청: 질병={patient.disease_code}, 중증도={patient.severity_code}, 위치=({patient.latitude}, {patient.longitude})")

        # 랭킹 계산
        ranked_hospitals = ranking_engine.rank_hospitals_by_location(
            patient=patient,
            top_n=10
        )

        # numpy/pandas 타입 변환
        ranked_hospitals = [_to_native(h) for h in ranked_hospitals]

        # 질환 정보 가져오기
        disease_info = DiseaseMapping.get_requirements(patient.disease_code)
        ktas_info = KTASMapping.get_level_info(patient.severity_code)

        return jsonify({
            'success': True,
            'patient': {
                'name': patient.name,
                'age': patient.age,
                'gender': patient.gender,
                'location': {
                    'latitude': patient.latitude,
                    'longitude': patient.longitude
                }
            },
            'disease_code': patient.disease_code,
            'disease_category': disease_info['category'],
            'disease_description': disease_info['description'],
            'severity_code': patient.severity_code,
            'severity_level': ktas_info['level'],
            'severity_name': ktas_info['name'],
            'total_ranked': len(ranked_hospitals),
            'ranked_hospitals': ranked_hospitals
        })

    except Exception as e:
        logger.error(f"랭킹 계산 오류: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/disease-codes', methods=['GET'])
def get_disease_codes():
    """사용 가능한 ICD-10 질병 코드 목록 조회
    ---
    tags:
      - Reference
    responses:
      200:
        description: ICD-10 코드 목록
    """
    codes = []
    for code, info in DiseaseMapping.ICD10_MAPPINGS.items():
        codes.append({
            'code': code,
            'category': info['category'],
            'description': info['description'],
            'required_services': info['service_names'],
            'requires_trauma_center': info.get('requires_trauma_center', False)
        })

    # 외상 코드 정보 추가
    codes.append({
        'code': 'S*, T*',
        'category': 'Trauma',
        'description': '중증 외상 (S00-T88)',
        'required_services': DiseaseMapping.TRAUMA_MAPPING['service_names'],
        'requires_trauma_center': True
    })

    return jsonify({'disease_codes': codes})


@app.route('/api/severity-codes', methods=['GET'])
def get_severity_codes():
    """KTAS 중증도 코드 목록 조회
    ---
    tags:
      - Reference
    responses:
      200:
        description: KTAS 중증도 코드 목록
    """
    codes = []
    for code, info in KTASMapping.LEVELS.items():
        codes.append({
            'code': code,
            'level': info['level'],
            'name': info['name'],
            'description': info['description']
        })

    return jsonify({'severity_codes': codes})


def initialize_server(data_dir: str = 'dataset'):
    """서버 초기화"""
    global ranking_engine

    logger.info("=" * 60)
    logger.info("EMS ML Server 초기화 시작 (v2.0 - ICD-10 기반)")
    logger.info("=" * 60)

    # 데이터 로드
    data_processor.load_and_process_data(data_dir)

    # 랭킹 엔진 초기화
    ranking_engine = HospitalRankingEngine(data_processor)

    logger.info("=" * 60)
    logger.info("EMS ML Server 초기화 완료")
    logger.info("=" * 60)


if __name__ == '__main__':
    # 서버 초기화
    initialize_server()

    # Flask 서버 실행
    app.run(
        host='0.0.0.0',
        port=25875,
        debug=True
    )

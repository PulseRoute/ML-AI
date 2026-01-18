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


from models import HospitalCandidate
from ems_mapping import EMSCodeMapping
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
        "description": "응급의료서비스(EMS) 전용 병원 추천 시스템 ML 서버 API",
        "version": "1.0.0",
        "contact": {
            "name": "EMS ML Team",
            "email": "ems-ml@example.com"
        }
    },
    "host": "localhost:5000",
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
            hospitals_loaded:
              type: integer
    """
    return jsonify({
        'status': 'healthy',
        'service': 'EMS ML Server',
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

    def _to_python(v):
        if v is None:
            return None
        
        if isinstance(v, (list, tuple, set)) or (hasattr(v, '__iter__') and not isinstance(v, (str, bytes))):
            try:
                iter(v)
                out = []
                for x in v:
                    out.append(_to_python(x))
                return out
            except Exception:
                pass

        try:
            if pd.isna(v):
                return None
        except Exception:
            pass

        if isinstance(v, (np.integer, np.floating, np.bool_)):
            return v.item()

        if hasattr(v, 'item') and not isinstance(v, (str, bytes)):
            try:
                return v.item()
            except Exception:
                pass

        return v

    for _, row in df.iterrows():
        entry = {
            'facid': _to_python(row.get('FACID')),
            'name': _to_python(row.get('FACNAME')),
            'address': _to_python(row.get('ADDRESS')),
            'city': _to_python(row.get('CITY')),
            'county': _to_python(row.get('COUNTY_NAME')),
            'latitude': _to_python(row.get('LATITUDE')),
            'longitude': _to_python(row.get('LONGITUDE')),
            'total_beds': _to_python(row.get('TOTAL_BEDS')) or 0,
            'service_codes': _to_python(row.get('service_codes')) or [],
            'service_names': _to_python(row.get('service_names')) or [],
            'has_trauma_center': _to_python(row.get('has_trauma_center')) or False,
            'cluster': _to_python(row.get('cluster'))
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
            - ems_code
            - triage_priority
            - hospital_candidates
          properties:
            ems_code:
              type: string
              enum: [STEMI, NSTEMI, Stroke, Trauma, Sepsis, Respiratory, General]
            triage_priority:
              type: integer
              minimum: 1
              maximum: 5
            hospital_candidates:
              type: array
              items:
                type: object
                properties:
                  facid:
                    type: string
                  duration:
                    type: number
                  distance:
                    type: number
    responses:
      200:
        description: 병원 랭킹 생성 성공
      400:
        description: 잘못된 요청
      500:
        description: 서버 에러
    """
    try:
        data = request.get_json()
        
        # 필수 파라미터 검증
        required_fields = ['ems_code', 'triage_priority', 'hospital_candidates']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        ems_code = data['ems_code']
        triage_priority = int(data['triage_priority'])
        candidates_data = data['hospital_candidates']
        
        # Validation
        if triage_priority < 1 or triage_priority > 5:
            return jsonify({'error': 'triage_priority must be between 1 and 5'}), 400
        
        if not candidates_data:
            return jsonify({'error': 'hospital_candidates cannot be empty'}), 400
        
        # HospitalCandidate 객체 생성
        candidates = [
            HospitalCandidate(
                facid=str(c['facid']),
                duration=float(c['duration']),
                distance=float(c['distance'])
            )
            for c in candidates_data
        ]
        
        logger.info(f"📍 랭킹 요청: EMS={ems_code}, Triage={triage_priority}, Candidates={len(candidates)}")
        
        # 랭킹 계산
        ranked_hospitals = ranking_engine.rank_hospitals(
            candidates=candidates,
            ems_code=ems_code,
            triage_priority=triage_priority,
            top_n=10
        )
        # Convert any numpy / pandas scalar types in ranked_hospitals to native Python types
        def _to_python(v):
          # Handle None
          if v is None:
            return None

          # Iterable (but not str/bytes) -> convert elements
          if isinstance(v, (list, tuple, set)) or (hasattr(v, '__iter__') and not isinstance(v, (str, bytes))):
            try:
              return [_to_python(x) for x in v]
            except Exception:
              pass

          # pandas/numpy NA
          try:
            if pd.isna(v):
              return None
          except Exception:
            pass

          # numpy scalar types
          if isinstance(v, (np.integer, np.floating, np.bool_)):
            return v.item()

          # objects with .item()
          if hasattr(v, 'item') and not isinstance(v, (str, bytes)):
            try:
              return v.item()
            except Exception:
              pass

          return v

        def _convert_obj(obj):
          if isinstance(obj, dict):
            return {k: _convert_obj(v) for k, v in obj.items()}
          if isinstance(obj, (list, tuple)):
            return [_convert_obj(x) for x in obj]
          return _to_python(obj)

        ranked_hospitals = [_convert_obj(h) for h in ranked_hospitals]
        
        return jsonify({
            'success': True,
            'ems_code': ems_code,
            'ems_description': EMSCodeMapping.get_requirements(ems_code)['description'],
            'triage_priority': triage_priority,
            'total_candidates': len(candidates),
            'total_capable': len(ranked_hospitals),
            'ranked_hospitals': ranked_hospitals
        })
        
    except Exception as e:
        logger.error(f"❌ 랭킹 계산 오류: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/ems-codes', methods=['GET'])
def get_ems_codes():
    """사용 가능한 EMS 코드 목록 조회
    ---
    tags:
      - Reference
    responses:
      200:
        description: EMS 코드 목록
    """
    codes = []
    for code, info in EMSCodeMapping.MAPPINGS.items():
        codes.append({
            'code': code,
            'description': info['description'],
            'required_services': info['service_names']
        })
    return jsonify({'ems_codes': codes})


def initialize_server(data_dir: str = 'dataset'):
    """서버 초기화"""
    global ranking_engine
    
    logger.info("=" * 60)
    logger.info("🚑 EMS ML Server 초기화 시작")
    logger.info("=" * 60)
    
    # 데이터 로드
    data_processor.load_and_process_data(data_dir)
    
    # 랭킹 엔진 초기화
    ranking_engine = HospitalRankingEngine(data_processor)
    
    logger.info("=" * 60)
    logger.info("✅ EMS ML Server 초기화 완료")
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

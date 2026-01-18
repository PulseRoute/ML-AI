from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, List, Optional
import os
from dataclasses import dataclass

# Project-specific imports
from models import PatientRequest
from disease_mapping import DiseaseMapping, KTASMapping
from data_processor import HospitalDataProcessor
from ranking import HospitalRankingEngine

warnings.filterwarnings('ignore')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- SWAGGER CONFIGURATION ---
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
        "title": "PulseRoute: EMS Hospital Recommendation ML API",
        "description": "An AI-powered emergency hospital recommendation system based on ICD-10 disease codes and KTAS (Korean Triage and Acuity Scale) severity levels.",
        "version": "2.0.0",
        "contact": {
            "name": "PulseRoute Team",
            "email": "star@devksy.xyz"
        }
    },
    "host": "localhost:25875",
    "basePath": "/",
    "schemes": ["http", "https"],
    "tags": [
        {"name": "Health", "description": "System status and health checks"},
        {"name": "Hospital Info", "description": "Accessing hospital metadata and facilities"},
        {"name": "Ranking", "description": "ML-driven hospital prioritization and ranking"},
        {"name": "Reference", "description": "Reference data for ICD-10 and Triage systems"}
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# Initialize engines
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
    """Server Health Check
    ---
    tags:
      - Health
    responses:
      200:
        description: Server is operational
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
        'service': 'PulseRoute ML Server',
        'version': '2.0.0',
        'hospitals_loaded': len(data_processor.hospital_data) if data_processor.hospital_data is not None else 0
    })


@app.route('/api/hospitals/info/<facid>', methods=['GET'])
def get_hospital_info(facid):
    """Get detailed information of a specific hospital
    ---
    tags:
      - Hospital Info
    parameters:
      - name: facid
        in: path
        type: string
        required: true
        description: Unique Facility Identifier (FACID)
    responses:
      200:
        description: Successfully retrieved hospital info
      404:
        description: Hospital not found
    """
    info = data_processor.get_hospital_info(facid)
    if info is None:
        return jsonify({'error': 'Hospital not found'}), 404

    result = _to_native(info)
    return jsonify(result)


@app.route('/api/hospitals/gach', methods=['GET'])
def get_gach_hospitals():
    """Retrieve list of hospitals supporting GACH
    ---
    tags:
      - Hospital Info
    responses:
      200:
        description: Returns a list of all hospitals in the system
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
    """Predict and rank best-fit hospitals for a patient
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
              description: "ICD-10 disease code (e.g., I21.3, S72.0)"
            severity_code:
              type: string
              enum: [KTAS_1, KTAS_2, KTAS_3, KTAS_4, KTAS_5]
              description: "Triage severity level (KTAS 1-5)"
            location:
              type: object
              properties:
                latitude:
                  type: number
                  description: Patient current latitude
                longitude:
                  type: number
                  description: Patient current longitude
    responses:
      200:
        description: Ranking successfully generated
      400:
        description: Invalid request body or missing fields
      500:
        description: Internal calculation error
    """
    try:
        data = request.get_json()

        # Parameter validation
        required_fields = ['age', 'gender', 'disease_code', 'severity_code', 'location']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        location = data.get('location', {})
        if 'latitude' not in location or 'longitude' not in location:
            return jsonify({'error': 'Missing latitude or longitude in location'}), 400

        # Create PatientRequest object
        patient = PatientRequest.from_dict(data)

        # Validate Severity Code
        if patient.severity_code not in KTASMapping.LEVELS:
            return jsonify({'error': f'Invalid severity_code. Must be one of: {list(KTASMapping.LEVELS.keys())}'}), 400

        logger.info(f"Ranking Req: Disease={patient.disease_code}, Severity={patient.severity_code}, Lat/Long=({patient.latitude}, {patient.longitude})")

        # Execute Ranking
        ranked_hospitals = ranking_engine.rank_hospitals_by_location(
            patient=patient,
            top_n=10
        )

        # Type conversion
        ranked_hospitals = [_to_native(h) for h in ranked_hospitals]

        # Fetch metadata
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
        logger.error(f"Ranking Error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/disease-codes', methods=['GET'])
def get_disease_codes():
    """Get list of supported ICD-10 disease codes
    ---
    tags:
      - Reference
    responses:
      200:
        description: List of ICD-10 categories and requirements
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

    # Add trauma codes
    codes.append({
        'code': 'S*, T*',
        'category': 'Trauma',
        'description': 'Severe Trauma (S00-T88)',
        'required_services': DiseaseMapping.TRAUMA_MAPPING['service_names'],
        'requires_trauma_center': True
    })

    return jsonify({'disease_codes': codes})


@app.route('/api/severity-codes', methods=['GET'])
def get_severity_codes():
    """Get KTAS Triage severity levels
    ---
    tags:
      - Reference
    responses:
      200:
        description: List of KTAS (Korean Triage and Acuity Scale) levels
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
    """Initialize server data and engines"""
    global ranking_engine

    logger.info("=" * 60)
    logger.info("Initializing PulseRoute ML Server (v2.0 - ICD-10 Support)")
    logger.info("=" * 60)

    # Data loading
    data_processor.load_and_process_data(data_dir)

    # Ranking engine setup
    ranking_engine = HospitalRankingEngine(data_processor)

    logger.info("=" * 60)
    logger.info("Server Initialization Complete")
    logger.info("=" * 60)


if __name__ == '__main__':
    initialize_server()

    # Launch Flask Server
    app.run(
        host='0.0.0.0',
        port=25875,
        debug=True
    )
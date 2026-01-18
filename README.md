# Hospital Recommendation Backend

## Project Overview

This project is a ML server for a hospital recommendation system for emergency patients.  
It provides RESTful APIs that suggest the most suitable hospital based on patient symptoms, severity, and candidate hospital information.

## Main Features

- Hospital detail lookup
- GACH (General Acute Care Hospital) list
- EMS code (emergency symptom classification) list
- Hospital ranking prediction based on patient status and candidate hospitals

## Folder Structure

```
app.py                # Main server entry point
data_processor.py     # Data processing and hospital info management
ems_mapping.py        # EMS code and service mapping
models.py             # Data model definitions
ranking.py            # Hospital ranking logic
requirements.txt      # Dependencies
dataset/              # Hospital/service/statistics CSV data
```

## How to Run

1. Install dependencies  
   ```
   pip install -r requirements.txt
   ```

2. Start the server  
   ```
   python app.py
   ```

3. Swagger docs:  
   [http://localhost:5000/swagger/](http://localhost:5000/swagger/)

## Example Usage

- Hospital info:  
  `GET /api/hospitals/info/<facid>`

- Hospital ranking prediction:  
  `POST /api/predict/rank`  
  (JSON body: ems_code, triage_priority, hospital_candidates)


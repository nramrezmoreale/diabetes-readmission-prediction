"""Send two example patients to the running service.

Usage:
    uv run python test_predict.py                      # http://localhost:9696
    uv run python test_predict.py http://<host>:<port>  # any other deployment
"""

import sys

import requests

url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696").rstrip("/")

low_risk_patient = {
    "race": "Caucasian", "gender": "Male", "age": "[50-60)",
    "admission_type_id": 3, "discharge_disposition_id": 1, "admission_source_id": 1,
    "time_in_hospital": 2, "medical_specialty": "Orthopedics-Reconstructive",
    "num_lab_procedures": 20, "num_procedures": 2, "num_medications": 10,
    "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
    "diag_1": "715", "diag_2": "250", "diag_3": "401", "number_diagnoses": 5,
    "A1Cresult": "None", "metformin": "Steady", "insulin": "No",
    "change": "No", "diabetesMed": "Yes",
}

high_risk_patient = {
    "race": "AfricanAmerican", "gender": "Female", "age": "[70-80)",
    "admission_type_id": 1, "discharge_disposition_id": 3, "admission_source_id": 7,
    "time_in_hospital": 8, "medical_specialty": "Nephrology",
    "num_lab_procedures": 60, "num_procedures": 1, "num_medications": 25,
    "number_outpatient": 2, "number_emergency": 3, "number_inpatient": 5,
    "diag_1": "428", "diag_2": "250.02", "diag_3": "585", "number_diagnoses": 9,
    "A1Cresult": ">8", "insulin": "Up", "change": "Ch", "diabetesMed": "Yes",
}

print("health:", requests.get(f"{url}/health", timeout=10).json())
for name, patient in [("low risk example", low_risk_patient), ("high risk example", high_risk_patient)]:
    response = requests.post(f"{url}/predict", json=patient, timeout=10)
    response.raise_for_status()
    print(f"{name}: {response.json()}")

"""Web service that predicts the risk of 30-day readmission.

Run locally:
    uv run uvicorn predict:app --host 0.0.0.0 --port 9696
Docs (Swagger UI): http://localhost:9696/docs
"""

import pickle
from typing import Literal

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

import features as F

MODEL_FILE = "model.bin"

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)
FEATURE_NAMES = list(dv.get_feature_names_out())

MedStatus = Literal["No", "Steady", "Up", "Down"]


class Patient(BaseModel):
    """One hospital encounter, using the same columns and codes as the UCI dataset."""

    race: str | None = Field(None, examples=["Caucasian"])
    gender: Literal["Female", "Male"] = "Female"
    age: str = Field("[70-80)", description="Age bracket, e.g. '[70-80)'", pattern=r"^\[\d+-\d+\)$")
    admission_type_id: int = Field(1, description="1=Emergency, 2=Urgent, 3=Elective, 7=Trauma")
    discharge_disposition_id: int = Field(1, description="1=Home, 3=SNF, 6=Home health ... see data/IDs_mapping.csv")
    admission_source_id: int = Field(7, description="7=Emergency room, 1=Physician referral ...")
    time_in_hospital: int = Field(3, ge=1, le=14, description="Length of stay in days")
    medical_specialty: str | None = Field(None, examples=["InternalMedicine"])
    num_lab_procedures: int = Field(40, ge=0)
    num_procedures: int = Field(1, ge=0)
    num_medications: int = Field(15, ge=0)
    number_outpatient: int = Field(0, ge=0, description="Outpatient visits in the previous year")
    number_emergency: int = Field(0, ge=0, description="Emergency visits in the previous year")
    number_inpatient: int = Field(0, ge=0, description="Inpatient visits in the previous year")
    diag_1: str | None = Field(None, description="Primary diagnosis (ICD-9)", examples=["428"])
    diag_2: str | None = Field(None, examples=["250.02"])
    diag_3: str | None = Field(None, examples=["401"])
    number_diagnoses: int = Field(7, ge=1)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    metformin: MedStatus = "No"
    repaglinide: MedStatus = "No"
    glimepiride: MedStatus = "No"
    glipizide: MedStatus = "No"
    glyburide: MedStatus = "No"
    pioglitazone: MedStatus = "No"
    rosiglitazone: MedStatus = "No"
    insulin: MedStatus = "No"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["No", "Yes"] = "No"


class Prediction(BaseModel):
    readmission_probability: float
    risk_level: Literal["low", "medium", "high"]
    high_risk: bool


def risk_level(p: float) -> str:
    if p < 0.10:
        return "low"
    if p < 0.20:
        return "medium"
    return "high"


def predict_single(patient: dict) -> float:
    raw = pd.DataFrame([patient])
    for med in F.ALL_MEDICATIONS:  # medications not in the API are "No"
        if med not in raw.columns:
            raw[med] = "No"
    features = F.prepare_features(raw)
    X = dv.transform(F.to_dicts(features))
    return float(model.predict(xgb.DMatrix(X, feature_names=FEATURE_NAMES))[0])


app = FastAPI(title="Diabetes 30-day readmission prediction")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient) -> Prediction:
    p = predict_single(patient.model_dump())
    level = risk_level(p)
    return Prediction(readmission_probability=round(p, 4), risk_level=level, high_risk=level == "high")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9696)

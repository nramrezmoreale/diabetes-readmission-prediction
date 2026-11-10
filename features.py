"""Shared data preparation for the diabetes readmission project.

Used by notebook.ipynb, train.py and predict.py so that training and serving
apply exactly the same transformations.
"""

import numpy as np
import pandas as pd

DATA_PATH = "data/diabetic_data.csv"

# Discharge codes meaning the patient died or went to hospice:
# such patients cannot be readmitted, so they are removed from training.
EXCLUDED_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

# Diabetes medications that are prescribed often enough to be useful
MEDICATIONS = [
    "metformin", "repaglinide", "glimepiride", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone", "insulin",
]
ALL_MEDICATIONS = MEDICATIONS + [
    "nateglinide", "chlorpropamide", "acetohexamide", "tolbutamide", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

TOP_SPECIALTIES = [
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
    "Cardiology", "Surgery-General", "Nephrology", "Orthopedics",
    "Orthopedics-Reconstructive", "Radiologist", "Pulmonology",
]

NUMERICAL = [
    "age_years", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "total_prior_visits",
    "num_med_changes",
]

CATEGORICAL = [
    "race", "gender", "admission_type", "discharge_disposition",
    "admission_source", "medical_specialty", "diag_1_group", "diag_2_group",
    "diag_3_group", "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + MEDICATIONS

FEATURES = NUMERICAL + CATEGORICAL


def icd9_group(code) -> str:
    """Map an ICD-9 diagnosis code to a broad disease group
    (grouping follows Strack et al., 2014)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "other"
    try:
        value = float(code)
    except ValueError:
        return "other"
    if int(value) == 250:
        return "diabetes"
    if 390 <= value <= 459 or int(value) == 785:
        return "circulatory"
    if 460 <= value <= 519 or int(value) == 786:
        return "respiratory"
    if 520 <= value <= 579 or int(value) == 787:
        return "digestive"
    if 800 <= value <= 999:
        return "injury"
    if 710 <= value <= 739:
        return "musculoskeletal"
    if 580 <= value <= 629 or int(value) == 788:
        return "genitourinary"
    if 140 <= value <= 239:
        return "neoplasms"
    return "other"


def map_admission_type(i) -> str:
    return {1: "emergency", 2: "urgent", 3: "elective", 7: "trauma"}.get(i, "other_or_unknown")


def map_discharge(i) -> str:
    if i == 1:
        return "home"
    if i in (6, 8):
        return "home_health"
    if i in (3, 4, 24):
        return "nursing_facility"
    if i in (2, 5, 9, 10, 15, 22, 23, 27, 28, 29, 30):
        return "other_facility"
    if i == 7:
        return "left_ama"
    return "other_or_unknown"


def map_admission_source(i) -> str:
    if i == 7:
        return "emergency_room"
    if i in (1, 2, 3):
        return "referral"
    if i in (4, 5, 6, 10, 18, 22, 25, 26):
        return "transfer"
    return "other_or_unknown"


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, na_values="?", low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the raw UCI data and create the binary target."""
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)]
    df = df[df["gender"] != "Unknown/Invalid"].copy()
    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
    return df.reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw UCI columns into model features. Works on the full dataset
    and on a single API request converted to a one-row DataFrame."""
    out = pd.DataFrame(index=df.index)

    # "[70-80)" -> 75
    out["age_years"] = df["age"].str.extract(r"\[(\d+)-")[0].astype(float) + 5

    for col in ["time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency",
                "number_inpatient", "number_diagnoses"]:
        out[col] = df[col].astype(float)
    out["total_prior_visits"] = (
        out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    )
    present_meds = [m for m in ALL_MEDICATIONS if m in df.columns]
    out["num_med_changes"] = df[present_meds].isin(["Up", "Down"]).sum(axis=1).astype(float)

    out["race"] = df["race"].fillna("Unknown")
    out["gender"] = df["gender"]
    out["admission_type"] = df["admission_type_id"].map(map_admission_type)
    out["discharge_disposition"] = df["discharge_disposition_id"].map(map_discharge)
    out["admission_source"] = df["admission_source_id"].map(map_admission_source)

    spec = df["medical_specialty"].fillna("Missing")
    out["medical_specialty"] = spec.where(spec.isin(TOP_SPECIALTIES + ["Missing"]), "Other")

    for i in (1, 2, 3):
        out[f"diag_{i}_group"] = df[f"diag_{i}"].map(icd9_group)

    # In recent pandas versions the string "None" is read as missing
    out["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    out["A1Cresult"] = df["A1Cresult"].fillna("None")
    out["change"] = df["change"]
    out["diabetesMed"] = df["diabetesMed"]
    for med in MEDICATIONS:
        out[med] = df[med]

    return out[FEATURES]


def to_dicts(features: pd.DataFrame) -> list[dict]:
    return features.to_dict(orient="records")

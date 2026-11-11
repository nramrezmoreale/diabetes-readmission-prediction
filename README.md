# Diabetes 30-day readmission prediction

Midterm project for [ML Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club.

## Problem description

When a patient is readmitted to hospital within 30 days of being discharged, it is usually a sign that something went wrong: the patient was sent home too early, did not understand the new medication, or had no follow-up. In the US, hospitals are financially penalised for excess 30-day readmissions, and every readmission means extra cost and extra risk for the patient. Diabetic patients are a large, high-risk group.

Hospitals can prevent part of these readmissions with follow-up actions: a phone call a few days after discharge, a medication review by a pharmacist, an early outpatient appointment. These resources are limited, so they should go to the patients who need them most.

**Goal:** at discharge, predict the probability that a diabetic patient will be readmitted within 30 days, so that care teams can rank patients by risk and prioritise follow-up.

**How the model is used:** the hospital system sends the encounter data (demographics, length of stay, number of prior visits, diagnoses, lab results, diabetes medication) to the web service at discharge. The service returns a readmission probability and a risk level (`low` / `medium` / `high`). Care coordinators get a daily list of high-risk patients to call.

### Dataset

[Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296) (UCI Machine Learning Repository; Strack et al., 2014, *Impact of HbA1c Measurement on Hospital Readmission Rates*).

- 101,766 hospital encounters of diabetic patients from 130 US hospitals, 50 columns
- Target: `readmitted` (`<30`, `>30`, `NO`) → binary `readmitted_30` (1 if readmitted within 30 days, ~11% of encounters)
- The dataset is committed in [`data/`](data/) (`diabetic_data.csv`, plus `IDs_mapping.csv` that explains the ID codes), so no download is needed.

### Approach

- Encounters where the patient died or went to hospice are removed (they cannot be readmitted).
- The data is split 60/20/20 **by patient**, so the same patient never appears in both training and evaluation data.
- Feature engineering ([`features.py`](features.py)): age bracket → number, ICD-9 diagnosis codes → 9 disease groups, ID codes → readable groups, total prior visits, number of medication dose changes.
- Classes are imbalanced, so the metric is **ROC AUC** (PR AUC also reported).

### Results

| Model | Best parameters | Validation ROC AUC | Validation PR AUC |
|---|---|---|---|
| Logistic regression | C = 0.001 | 0.654 | 0.195 |
| Decision tree | max_depth = 7, min_samples_leaf = 200 | 0.656 | 0.192 |
| Random forest | 200 trees, max_depth = 15, min_samples_leaf = 5 | 0.663 | 0.204 |
| **XGBoost** | eta = 0.05, max_depth = 6, min_child_weight = 30, 207 rounds | **0.668** | **0.205** |

The final XGBoost model scores **ROC AUC 0.670 and PR AUC 0.217 on the test set** (a random model would get 0.5 and 0.114). This is in line with published results on this dataset: readmission is hard to predict from encounter data alone, but the model is useful for ranking. For example, the 20% of patients with the highest scores contain about 40% of all 30-day readmissions.

The strongest predictors are the number of inpatient visits in the previous year, the discharge destination (home vs. nursing or other facility) and the number of emergency visits.

## Project structure

```
├── data/
│   ├── diabetic_data.csv     # dataset (UCI)
│   └── IDs_mapping.csv       # meaning of admission/discharge/source IDs
├── notebook.ipynb            # data cleaning, EDA, feature importance, model selection and tuning
├── features.py               # feature engineering shared by notebook, training and service
├── train.py                  # trains the final model and saves model.bin
├── model.bin                 # trained model (DictVectorizer + XGBoost)
├── predict.py                # FastAPI web service
├── test_predict.py           # sends example requests to the service
├── pyproject.toml, uv.lock   # dependencies (uv)
├── Dockerfile
├── deploy-kind.sh            # deploys to a local Kubernetes cluster (kind)
├── k8s/                      # Kubernetes manifests
└── images/                   # screenshots of the deployed service
```

## How to run

### 1. Install dependencies

The project uses [uv](https://docs.astral.sh/uv/) for dependency and virtual environment management (Python 3.12).

```bash
# install uv if you don't have it
pip install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/nramrezmoreale/diabetes-readmission-prediction.git
cd diabetes-readmission-prediction

uv sync                   # creates .venv and installs all dependencies from uv.lock
source .venv/bin/activate # activate the virtual environment (Windows: .venv\Scripts\activate)
```

With the environment activated you can run `python ...` directly; without activating it, prefix commands with `uv run`.

### 2. Notebook

```bash
uv run jupyter notebook notebook.ipynb
```

### 3. Train the model

```bash
uv run python train.py
```

Output:

```
Training on 79532 rows, testing on 19808 rows...
Test ROC AUC: 0.6698
Test PR AUC:  0.2168
Training the final model on all data...
Model saved to model.bin
```

### 4. Run the web service locally

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

Endpoints:
- `GET /health` – health check
- `POST /predict` – returns the readmission probability for one patient
- `GET /docs` – interactive Swagger UI where you can try requests in the browser

In another terminal:

```bash
uv run python test_predict.py
```

```
health: {'status': 'ok'}
low risk example: {'readmission_probability': 0.0287, 'risk_level': 'low', 'high_risk': False}
high risk example: {'readmission_probability': 0.4026, 'risk_level': 'high', 'high_risk': True}
```

Or with curl:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "age": "[70-80)",
    "admission_type_id": 1, "discharge_disposition_id": 3, "admission_source_id": 7,
    "time_in_hospital": 8, "medical_specialty": "Nephrology",
    "num_lab_procedures": 60, "num_procedures": 1, "num_medications": 25,
    "number_outpatient": 2, "number_emergency": 3, "number_inpatient": 5,
    "diag_1": "428", "diag_2": "250.02", "diag_3": "585", "number_diagnoses": 9,
    "A1Cresult": ">8", "insulin": "Up", "change": "Ch", "diabetesMed": "Yes"
  }'
```

All fields use the same names and codes as the original dataset; any field left out takes a default value (see the `Patient` model in `predict.py` or `/docs`). Risk levels: `low` < 0.10 ≤ `medium` < 0.20 ≤ `high`.

### 5. Run with Docker

```bash
docker build -t diabetes-readmission:v1 .
docker run -it --rm -p 9696:9696 diabetes-readmission:v1
```

Then test it with `uv run python test_predict.py` (or open http://localhost:9696/docs).

### 6. Deploy to Kubernetes (kind)

Requirements: [Docker](https://docs.docker.com/get-docker/), [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) and [kubectl](https://kubernetes.io/docs/tasks/tools/).

```bash
./deploy-kind.sh
```

The script builds the image, creates a kind cluster (`k8s/kind-config.yaml`), loads the image into it and applies [`k8s/deployment.yaml`](k8s/deployment.yaml) (2 replicas with health probes) and [`k8s/service.yaml`](k8s/service.yaml) (NodePort exposed on port 30080).

Test the deployed service:

```bash
uv run python test_predict.py http://localhost:30080
```

![Service running on Kubernetes](images/kubernetes-test.png)

Clean up:

```bash
kind delete cluster --name readmission
```

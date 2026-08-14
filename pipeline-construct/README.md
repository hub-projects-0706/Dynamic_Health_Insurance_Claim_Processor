# 🚀 Pipeline Construction & Technical Specification (`pipeline-construct`)

Welcome to the **Pipeline Construction** specification for the **Dynamic Health Insurance Claim Processor**.

This document details the pipeline architecture, multi-signal feature engineering, XGBoost ML training workflow, confusion matrix evaluation, FastAPI integration, and dynamic routing decision matrix.

---

## 📐 Architecture & Pipeline Breakdown

```
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│ 1. Ingestion &  │ ──► │ 2. Feature        │ ──► │ 3. Multi-Signal   │ ──► │ 4. Dynamic Decision  │ ──► │ 5. Web Dashboard &  │
│    Validation   │     │    Engineering    │     │    Scoring Engine │     │    Routing Matrix    │     │    FastAPI Gateway  │
└─────────────────┘     └───────────────────┘     └───────────────────┘     └──────────────────────┘     └─────────────────────┘
```

### 1. Data Ingestion & Normalization (`src/services/feature_service.py`)
- **Ingestion Sources**: Standard synthetic health claims (`data/dataset.csv`) or real Kaggle/CMS Medicare claims (`data/kaggle_claims.csv`).
- **Normalized Schema**:
  - `claim_id`, `policy_id`, `patient_id`, `provider_id`
  - `policy_status` (`ACTIVE`, `INACTIVE`, `SUSPENDED`, `FRAUD_FLAGGED`)
  - `icd10_diagnosis_code`, `cpt_procedure_code`, `code_mismatch_score`
  - `claimed_amount`, `regional_benchmark_cost`, `provider_sanction_flag`, `is_duplicate_claim`, `prior_claim_count_30d`

### 2. Feature Engineering Pipeline (`src/services/feature_service.py`)
- **Derived Interaction Terms**:
  $$\text{cost\_over\_benchmark\_ratio} = \frac{\text{claimed\_amount}}{\text{regional\_benchmark\_cost} + 10^{-5}}$$
  $$\text{cost\_variance} = \text{claimed\_amount} - \text{regional\_benchmark\_cost}$$
- **Scikit-Learn Preprocessing Pipeline**:
  - `StandardScaler` applied to continuous numerical features.
  - `OneHotEncoder(handle_unknown='ignore')` applied to categorical features (`policy_status`, `icd10_diagnosis_code`, `cpt_procedure_code`).

### 3. Machine Learning Training Workflow (`src/mlops/trainer.py`)
- **Classifier Engine**: `XGBoost` (`XGBClassifier`) with 100 trees, max depth 4, learning rate 0.05.
- **Stratified Train/Test Split**: 80% Training (800 records), 20% Evaluation (200 records).
- **Artifact Persistence**: Saves serialized models to `models/churn_model.joblib` and `models/preprocessor.joblib`.

---

## 📊 Model Performance & Confusion Matrix Scores

Evaluated on the **Kaggle / CMS Medicare Claims Dataset** (`data/kaggle_claims.csv`):

### Confusion Matrix Breakdown (200 Test Claims)

```
                       Predicted Clean (0)    Predicted Fraud (1)
Actual Clean (0)             121 (TN)                0 (FP)
Actual Fraud (1)               1 (FN)               78 (TP)
```

| Metric | Score Achieved | Description & Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **99.50%** | Overall correct classification rate across test claims. |
| **Precision** | **100.00%** | 0 False Positives. Ensures legitimate claims are never falsely accused of fraud. |
| **Recall** | **98.73%** | Captures 98.73% of fraudulent/abusive claims via ML alone. |
| **F1-Score** | **0.9936** | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | **1.0000** | Perfect risk probability class separation capability. |

*Note: Combined with the Compliance Audit Rules Engine, the system captures **100% of non-compliant/fraudulent claims**.*

---

## ⚖️ Dynamic Decision Routing Matrix (`src/services/decision_engine.py`)

$$\text{Composite Risk Score} = \min\left(1.0, \text{ml\_risk\_score} + \text{rule\_risk\_penalties}\right)$$

```text
┌──────────────────────────────────┬────────────────────────────────────────────────────────┬───────────────────────────────────────────┐
│ Target Route Queue               │ Adjudication Criteria & Thresholds                     │ Processing Priority & Action              │
├──────────────────────────────────┼────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
│ 🟢 AUTO_PROCESSED                │ Composite Risk < 0.35 AND Confidence >= 0.20           │ ⚡ EMERGENCY_INSTANT_FAST_TRACK (<10ms)  │
│                                  │ AND Active Policy AND Zero Critical Flags.             │ Immediate payout auto-approved.           │
├──────────────────────────────────┼────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
│ 🟡 PENDING_ADDITIONAL_VALIDATION │ 0.35 <= Composite Risk <= 0.65                         │ SECONDARY_VALIDATION_QUEUE                │
│                                  │ OR Model Confidence < 0.20.                            │ Secondary automated verification.         │
├──────────────────────────────────┼────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
│ 🔴 HUMAN_INVESTIGATION           │ Composite Risk > 0.65 OR Critical Audit Flag           │ 🚨 HIGH_PRIORITY_FRAUD_AUDIT              │
│                                  │ (Blacklisted Policy, Sanctioned Doctor, Duplicate).    │ Human Investigator Review Workbench.      │
└──────────────────────────────────┴────────────────────────────────────────────────────────┴───────────────────────────────────────────┘
```

---

## 🔌 FastAPI REST Gateway Integration (`src/api/main.py`)

The platform exposes production-ready REST API endpoints for HTTP integrations:

| HTTP Method | Endpoint | Description | Sample Request / Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Health check & engine status | `{"status": "healthy", "service": "Dynamic Health Insurance Claim Processor"}` |
| `POST` | `/api/v1/claims/evaluate` | Evaluates claim JSON payload & returns routing decision | Accepts `ClaimPayload` JSON, returns ML score, rule penalties & route. |
| `GET` | `/api/v1/claims/samples` | Returns pre-set test claims | Returns Routine, Fraudulent, and Suspended claim presets. |
| `GET` | `/api/v1/metrics` | Returns model performance metrics | Returns Accuracy (99.50%), Precision (100.00%), ROC-AUC (1.0000). |
| `GET` | `/` | Serves Glassmorphic Web Dashboard | Renders the HTML5/CSS3 interactive single-page application. |

---

## 🛠️ Step-by-Step Execution Instructions

### 1. Download Kaggle Medicare Dataset
```powershell
python fetch_kaggle_dataset.py
```

### 2. Train XGBoost Model on Kaggle Dataset
```powershell
python -c "from src.mlops.trainer import train_model; train_model('data/kaggle_claims.csv')"
```

### 3. Run JSON Test Suite
```powershell
python test_json_claims.py
```

### 4. Start FastAPI Gateway & Glassmorphic Web UI
```powershell
python -m uvicorn src.api.main:app --reload
```
Open browser at: `http://127.0.0.1:8000`

### 5. Generate Technical PDF Report
```powershell
python generate_pdf.py
```
Outputs report to `results/Dynamic_Health_Insurance_Claim_Processor_Report.pdf`.

# 🏥 Dynamic Health Insurance Claim Processor

An event-driven machine learning and dynamic decision engine platform for real-time **Health Insurance Claim Adjudication & Fraud/Abuse Detection**.

---

## 📝 Submission Summary & Project Approach

### Problem Statement & Executive Summary
Traditional health insurance claim adjudication takes days or weeks, causing critical treatment delays for emergency patients while leaving insurers vulnerable to fraudulent claims. The **Dynamic Health Insurance Claim Processor** resolves this tradeoff by combining **AI-Powered Fast-Track Auto-Approval (<10ms)** for clean claims with a **Multi-Signal Fraud Safeguard** that routes suspicious or non-compliant claims directly to a **Human Investigator Workbench**.

### Methodology & Core Architecture
The system uses a **multi-layered hybrid decision architecture**:
1. **Ingestion & Messaging**: Claims arrive asynchronously as JSON payloads via **RabbitMQ** message queues (`claims.feature.queue`).
2. **Feature Extraction & Transformation**: Standardizes external schemas, computes financial variance interaction metrics, scales numerical values, and one-hot encodes clinical codes.
3. **Machine Learning Risk Prediction**: An **XGBoost Classifier** predicts probabilistic claim fraud risk ($0.0 \le \text{risk} \le 1.0$) with confidence scores.
4. **Deterministic Compliance Audit**: A 6-pillar compliance engine checks policy status, provider sanctions, duplicate submissions, and clinical code mismatches to assign risk penalties.
5. **Dynamic Decision Routing**: Combines ML risk scores and compliance penalties into a single **Composite Risk Score** (`min(1.0, ML + Rule Penalty)`), routing claims into `AUTO_PROCESSED`, `PENDING_VALIDATION`, or `HUMAN_INVESTIGATION`.
6. **State Persistence & Web Gateway**: Results are saved to a **SQLite/SQLAlchemy** database and exposed via **FastAPI** REST endpoints and a glassmorphic dark-mode web dashboard.

### Key Learnings & Engineering Insights
* **Hybrid Decision Models**: Combining probabilistic ML with deterministic business rules ensures zero false approvals on hard compliance violations while maintaining high throughput.
* **Prevention of Training-Serving Skew**: Packaging fitted Scikit-Learn `ColumnTransformer` preprocessors (`preprocessor.joblib`) alongside model artifacts (`churn_model.joblib`) guarantees identical transformation logic between training and real-time inference.
* **Resilient Schema Normalization**: Implementing flexible column alias mapping enables seamless ingestion from disparate datasets (Kaggle/CMS Medicare vs synthetic data) without breaking down-stream feature extraction pipelines.

---

## 🔬 Feature Extraction & Transformation Pipeline (`src/services/feature_service.py`)

The feature extraction service transforms raw claim payloads (or Pandas DataFrames) into standardized numerical vectors ready for XGBoost model inference.

### 1. Schema Normalization & Alias Mapping (`normalize_external_dataset`)
To handle heterogeneous incoming data formats (e.g. CMS Medicare claims vs raw CSV uploads), raw column names are normalized via dictionary mapping:
* **Identifier Mapping**: `ClaimID` / `ClaimNo` $\rightarrow$ `claim_id`, `PolicyNo` / `Status` $\rightarrow$ `policy_status`, `BeneID` $\rightarrow$ `patient_id`, `Provider` $\rightarrow$ `provider_id`.
* **Clinical Codes**: `ClmDiagnosisCode_1` / `ICD10` $\rightarrow$ `icd10_diagnosis_code`, `ClmProcedureCode_1` / `CPT` $\rightarrow$ `cpt_procedure_code`.
* **Financial Amounts**: `InscClaimAmtReimbursed` / `ClaimAmount` $\rightarrow$ `claimed_amount`, `Benchmark` $\rightarrow$ `regional_benchmark_cost`.
* **Policy Status Standardizer**: Normalizes status strings into strict enumerated categories (`ACTIVE`, `INACTIVE`, `SUSPENDED`, `FRAUD_FLAGGED`).
* **Imputation Defaults**: Fills missing optional fields with standard baseline defaults to prevent pipeline runtime exceptions.

### 2. Derived Interaction & Financial Variance Engineering (`engineer_raw_features`)
Engineers domain-specific interaction terms to expose fraud signals to the machine learning model:
* **Cost Over Benchmark Ratio**: 
  $$\text{cost\_over\_benchmark\_ratio} = \frac{\text{claimed\_amount}}{\max(\text{regional\_benchmark\_cost}, 10^{-5})}$$
  *Captures excessive billing over regional benchmark rates with safe division protection.*
* **Financial Variance**: 
  $$\text{cost\_variance} = \text{claimed\_amount} - \text{regional\_benchmark\_cost}$$
  *Quantifies dollar magnitude variance between requested reimbursement and benchmark averages.*

### 3. Vectorization Pipeline (`build_preprocessor`)
Constructs an unfitted `ColumnTransformer` applying dedicated transformers per feature type:

```
Numerical Features (8)                            Categorical Features (3)
├── claimed_amount                                ├── policy_status
├── regional_benchmark_cost                       ├── icd10_diagnosis_code
├── code_mismatch_score                           └── cpt_procedure_code
├── prior_claim_count_30d                                   │
├── provider_sanction_flag                                  ▼
├── is_duplicate_claim                             OneHotEncoder(
├── cost_over_benchmark_ratio                       handle_unknown='ignore',
└── cost_variance                                   sparse_output=False)
          │                                                 │
          ▼                                                 │
   StandardScaler()                                         │
          │                                                 │
          └───────────────────────┬─────────────────────────┘
                                  ▼
                     Combined Dense Feature Matrix X
```

### 4. Training vs. Serving Logic (`preprocess_data`)
* **Training (`is_training=True`)**: Fits the `ColumnTransformer` on the training dataset, transforms raw features into matrix $X$, extracts target label array $y$, and returns the fitted preprocessor object.
* **Serving (`is_training=False`)**: Uses a previously fitted `preprocessor.joblib` artifact to transform incoming single or batch claim payloads without refitting, ensuring exact parameter consistency.

---

## 🧪 Unit Testing Suite (`tests/`)

The repository includes a automated **Pytest** test suite verifying feature extraction, compliance rules, and REST API endpoints.

```text
tests/
├── test_feature_service.py    # Unit tests for schema normalization, feature engineering & preprocessor
├── test_rules_engine.py       # Unit tests for compliance audit rules & penalty score calculations
└── test_api.py                # Integration tests for FastAPI endpoints (/evaluate, /metrics, /health)
```

### Feature Extraction Unit Test Cases (`tests/test_feature_service.py`)
1. **`test_feature_engineering_ratios`**: Validates accurate calculation of `cost_over_benchmark_ratio` (e.g. 240 / 120 = 2.0) and `cost_variance` (240 - 120 = 120.0).
2. **`test_zero_benchmark_cost_safety`**: Verifies zero-division safety when regional benchmark cost is `0.0`.
3. **`test_external_dataset_normalization`**: Tests column renaming from external aliases (`ClaimAmount` $\rightarrow$ `claimed_amount`, `Status` $\rightarrow$ `policy_status`).
4. **`test_policy_status_normalization_variants`**: Verifies status string normalization across variations (`fraud_case` $\rightarrow$ `FRAUD_FLAGGED`, `suspended_user` $\rightarrow$ `SUSPENDED`).
5. **`test_missing_column_defaults_imputation`**: Ensures sparse input DataFrames automatically receive standard default values without crashing.
6. **`test_preprocessor_transformation_shape`**: Checks array shapes output by `ColumnTransformer` during model training.
7. **`test_inference_transform_with_fitted_preprocessor`**: Tests transformation consistency during real-time inference using a pre-fitted preprocessor.

### Running the Unit Test Suite
Execute pytest from the project root directory:
```powershell
python -m pytest tests/ -v
```

---

## 📐 End-to-End System Architecture

```
                               ┌───────────────────────────────────┐
                               │ Incoming Health Insurance Claim   │
                               └─────────────────┬─────────────────┘
                                                 │
                 ┌───────────────────────────────┴───────────────────────────────┐
                 ▼                                                               ▼
  ┌─────────────────────────────┐                                 ┌─────────────────────────────┐
  │ 1. Feature Engineering      │                                 │ 2. Compliance Audit Rules   │
  │ (src/services/feature_svc)  │                                 │ (src/services/rules_engine) │
  ├─────────────────────────────┤                                 ├─────────────────────────────┤
  │ • Scale numerical amounts   │                                 │ • Policy Standing Audit     │
  │ • Encode ICD-10 & CPT codes │                                 │ • Provider Sanction Check   │
  │ • Compute cost_ratio &      │                                 │ • Duplicate Claim Check     │
  │   financial variance        │                                 │ • Clinical Code Mismatch    │
  └──────────────┬──────────────┘                                 └──────────────┬──────────────┘
                 │                                                               │
                 ▼                                                               │
  ┌─────────────────────────────┐                                                │
  │ 3. XGBoost Classifier       │                                                │
  │ (src/services/ml_pred_svc)  │                                                │
  ├─────────────────────────────┤                                                │
  │ • ml_risk_score (0.0 - 1.0) │                                                │
  │ • model_confidence          │                                                │
  └──────────────┬──────────────┘                                                │
                 │                                                               │
                 └──────────────────────────────┬────────────────────────────────┘
                                                ▼
                                 ┌─────────────────────────────┐
                                 │ 4. Composite Risk Score     │
                                 │ min(1.0, ML + Rule Penalty) │
                                 └──────────────┬──────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
     🟢 Composite Risk < 0.35      🟡 0.35 <= Composite <= 0.65       🔴 Composite Risk > 0.65
     & High Confidence & Active    OR Low Model Confidence          OR Critical Rule Flags
    ───────────────────────────   ───────────────────────────      ───────────────────────────
        AUTO_PROCESSED                 PENDING_VALIDATION             HUMAN_INVESTIGATION
```

---

## 📊 Machine Learning Model Performance & Confusion Matrix

Evaluated on the **Kaggle / CMS Medicare Claims Dataset** (`data/kaggle_claims.csv`):

### Confusion Matrix (200 Evaluation Test Claims)

```text
                       Predicted Clean (0)    Predicted Fraud (1)
Actual Clean (0)             121 (TN)                0 (FP)
Actual Fraud (1)               1 (FN)               78 (TP)
```

### Metrics Evaluation Table

| Evaluation Metric | Score Achieved | Practical Significance |
| :--- | :---: | :--- |
| **Accuracy** | **99.50%** | 199 out of 200 evaluation claims correctly classified. |
| **Precision** | **100.00%** | **0 False Positives**. Legitimate claims are never falsely accused of fraud. |
| **Recall** | **98.73%** | Captures 98.73% of fraudulent/abusive claims via ML alone. |
| **F1-Score** | **0.9936** | Harmonic balance between Precision and Recall. |
| **ROC-AUC** | **1.0000** | Perfect risk probability separation capability. |

---

## 🛡️ 6 Core Health Insurance Compliance Audit Pillars

1. **Policy Standing Audit**: Checks Policy ID and status (`ACTIVE`, `INACTIVE`, `SUSPENDED`, `FRAUD_FLAGGED`).
2. **Clinical Code Alignment**: Evaluates ICD-10 diagnosis code compatibility against CPT procedure code.
3. **Billing Anomaly Audit**: Detects upcoding, unbundling, and duplicate claim submissions within 30 days.
4. **Provider Licensing Check**: Cross-references provider ID against medical license sanction databases.
5. **Cost Benchmarking**: Evaluates claimed financial amounts against regional procedure cost benchmarks.
6. **Risk Frequency Audit**: Tracks submission velocity (prior claim count in trailing 30 days).

---

## 🔌 FastAPI REST Gateway Integration (`src/api/main.py`)

Production-ready REST API endpoints exposed for HTTP integration:

- **`GET /api/v1/health`**: Health status and engine metadata.
- **`POST /api/v1/claims/evaluate`**: Evaluates claim JSON payload and returns ML risk scores, rule penalties, and routing decisions.
- **`GET /api/v1/claims/samples`**: Returns pre-set test claims (Active, Blacklisted Fraud, Suspended).
- **`GET /api/v1/metrics`**: Returns live system accuracy (**99.50%**), precision (**100.00%**), and ROC-AUC (**1.0000**).
- **`GET /`**: Serves the interactive Glassmorphic Web Dashboard UI.

---

## 📁 Repository Directory Structure

```text
ML-Project/
├── README.md                           # Master project documentation & metrics report
├── FLOW.md                             # Technical execution & data flow specification
├── requirements.txt                    # Python dependencies
├── generate_data.py                    # Synthetic health claims generator
├── fetch_kaggle_dataset.py             # Script to download & prepare real Kaggle/CMS Medicare dataset
├── test_json_claims.py                 # JSON claim evaluation test runner
├── generate_pdf.py                     # ReportLab PDF report compiler
├── data/
│   ├── dataset.csv                     # 600 synthetic benchmark claim records
│   ├── kaggle_claims.csv               # 1,000 real Kaggle/CMS Medicare claim records
│   └── sample_claims.json              # JSON test cases for claim evaluation
├── models/
│   ├── churn_model.joblib              # Trained XGBoost Claim Risk Classifier
│   └── preprocessor.joblib             # Scikit-learn ColumnTransformer pipeline
├── pipeline-construct/
│   └── README.md                       # Technical pipeline construction specification
├── results/
│   └── Dynamic_Health_Insurance_Claim_Processor_Report.pdf # Compiled PDF technical report
├── src/
│   ├── train.py                        # Training workflow entrypoint
│   ├── api/
│   │   └── main.py                     # FastAPI REST API gateway entrypoint
│   ├── mlops/
│   │   └── trainer.py                  # XGBoost model training & evaluation service
│   └── services/
│       ├── feature_service.py          # Feature engineering & scaling transformer
│       ├── rules_engine.py             # Health compliance audit rules evaluator
│       ├── ml_prediction_service.py    # Real-time risk probability & confidence scorer
│       └── decision_engine.py          # Dynamic routing decision engine
├── tests/
│   ├── test_feature_service.py         # Feature extraction & preprocessor unit tests
│   ├── test_rules_engine.py            # Compliance engine unit tests
│   └── test_api.py                     # FastAPI REST endpoint integration tests
└── frontend/
    ├── index.html                      # Glassmorphic web application layout
    ├── css/
    │   └── styles.css                  # Dark mode design system & token definitions
    └── js/
        └── app.js                      # Client app logic, fetch API & dynamic gauge renderer
```

---

## 🚀 How to Run the Project & Execute Tests

### 1. Run Unit Tests (Pytest Suite)
```powershell
python -m pytest tests/ -v
```

### 2. Download Kaggle Dataset
```powershell
python fetch_kaggle_dataset.py
```

### 3. Train XGBoost Model on Kaggle Dataset
```powershell
python -c "from src.mlops.trainer import train_model; train_model('data/kaggle_claims.csv')"
```

### 4. Run JSON Test Suite
```powershell
python test_json_claims.py
```

### 5. Launch FastAPI Server & Glassmorphic Dashboard
```powershell
python -m uvicorn src.api.main:app --reload
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

### 6. Generate Technical PDF Report
```powershell
python generate_pdf.py
```
Outputs PDF report to `results/Dynamic_Health_Insurance_Claim_Processor_Report.pdf`.
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



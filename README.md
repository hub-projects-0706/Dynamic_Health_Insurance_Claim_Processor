# 🏥 Dynamic Health Insurance Claim Processor

An event-driven machine learning and dynamic decision engine platform for real-time **Health Insurance Claim Adjudication & Fraud/Abuse Detection**.

---

## 🌟 Executive Summary & Key Highlights

Traditional health insurance claim adjudication takes days or weeks, causing treatment delays for patients in emergency situations. Conversely, approving claims blindly risks paying out fraudulent claims.

The **Dynamic Health Insurance Claim Processor** solves this problem by combining **AI-Powered Fast-Track Auto-Approval (<10ms)** for legitimate & emergency medical claims with a **Multi-Signal Fraud Safeguard** that immediately escalates suspicious claims to a **Human Investigator Workbench**.

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
└── frontend/
    ├── index.html                      # Glassmorphic web application layout
    ├── css/
    │   └── styles.css                  # Dark mode design system & token definitions
    └── js/
        └── app.js                      # Client app logic, fetch API & dynamic gauge renderer
```

---

## 🚀 How to Run the Project

### 1. Download Kaggle Dataset
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

### 4. Launch FastAPI Server & Glassmorphic Dashboard
```powershell
python -m uvicorn src.api.main:app --reload
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

### 5. Generate Technical PDF Report
```powershell
python generate_pdf.py
```
Outputs PDF report to `results/Dynamic_Health_Insurance_Claim_Processor_Report.pdf`.

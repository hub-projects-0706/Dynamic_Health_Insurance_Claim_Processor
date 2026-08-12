# Adaptive Health Insurance Claim Intelligence & Dynamic Routing Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM-orange.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Tuning-Optuna-blue.svg)](https://optuna.org/)
[![RabbitMQ](https://img.shields.io/badge/Messaging-RabbitMQ-FF6600.svg)](https://www.rabbitmq.com/)
[![Redis](https://img.shields.io/badge/Cache-Redis-DC382D.svg)](https://redis.io/)
[![Prometheus](https://img.shields.io/badge/Monitoring-Prometheus-E6522C.svg)](https://prometheus.io/)

An event-driven, production-oriented health insurance claim intelligence platform that leverages **Machine Learning (XGBoost/LightGBM)**, **Historical Claim Similarity Search**, **Configurable Business Rules**, **CQRS Architecture**, and **Human-in-the-Loop MLOps** to dynamically route claims into the safest and most efficient processing path.

---

## 🎯 Executive Overview & Problem Statement

Health insurance companies process massive volumes of claims daily. Treating every claim through a uniform, rigid workflow introduces high operational costs, unnecessary manual effort, and settlement delays.

The platform answers a fundamental operational question:
> **“What is the safest and most efficient processing path for this particular claim?”**

Instead of relying on isolated binary fraud prediction, the platform combines multiple intelligence signals—ML risk probability, model confidence, historical claim similarity, member/provider behavioral history, procedure complexity, and business rules—to dynamically route each claim:

```text
Low Risk + High Confidence + Clean History       ──►  ⚡ Automatic Processing
Medium Risk  OR  Low Confidence  OR  Rule Warning ──►  🔍 Additional Validation
High Risk + High Confidence  OR  Sanction Match   ──►  🕵️ Human Investigation
```

---


---

## 💡 Key Platform Capabilities

1. **Multi-Signal Intelligence Engine**:
   - **ML Risk Prediction**: XGBoost / LightGBM models trained on claim, diagnosis, procedure, member, and provider feature vectors.
   - **Model Confidence**: Statistical confidence estimation derived from tree variance and probability margin.
   - **Historical Claim Similarity**: Nearest-neighbor vector similarity matching over historical claim outcomes (% auto-approved, % flagged, % confirmed fraud).
   - **Configurable Business Rules Engine**: Hard business constraint checks (provider sanction status, high-value threshold, rapid multi-claim window, diagnosis-procedure alignment).

2. **Dynamic Decision Routing**:
   - **Auto Processing**: Simple, routine, low-risk, high-confidence claims automatically approved without manual intervention.
   - **Additional Validation**: Claims with minor rule warnings, medium risk scores, or low model confidence routed for targeted automated or semi-automated checks.
   - **Human Investigation**: Suspicious, high-risk, high-value, or anomalous claims escalated directly to the human investigator queue.

3. **CQRS Architecture (Command Query Responsibility Segregation)**:
   - **Commands**: Submit Claim, Update Claim, Process Claim, Approve Claim, Reject Claim, Escalate Claim, Submit Investigator Feedback.
   - **Queries**: Get Claim Status, Get Risk Breakdown & SHAP Features, Get Historical Similar Claims, Get Investigator Queue, Get MLOps & Retraining Logs, Get System Metrics.

4. **Human-in-the-Loop Learning & MLOps**:
   - Investigator decisions (Approved, Rejected, Confirmed Fraud) are persisted as ground-truth feedback.
   - Retraining pipeline uses **Optuna** for hyperparameter optimization and **MLflow** for experiment tracking and champion model registration.
   - Real-time **Drift Monitoring** (KS-test, Population Stability Index) to catch covariate and prediction drift.

5. **Observability & Prometheus Monitoring**:
   - **System Metrics**: Throughput (claims/sec), API latency, RabbitMQ queue depth, error rates.
   - **ML Metrics**: Prediction distribution, Model Precision/Recall, ROC-AUC, False Positives/Negatives, Model Confidence scores, Data Drift indices.

---



## 🛠️ Main System Components

| Component ID | Component Name | Description |
|---|---|---|
| **1** | **Claim Ingestion Service** | Accepts incoming claims via REST/CQRS commands and pushes `ClaimSubmitted` events. |
| **2** | **Claim Validation Service** | Checks schema completeness, ICD-10/CPT validity, member policy status, and coverage bounds. |
| **3** | **Data Processing Pipeline** | Normalizes claim payloads, handles missing values, and prepares clean records. |
| **4** | **Feature Engineering Service** | Computes operational ratios, provider risk rates, member 30d frequencies, and procedure complexity. |
| **5** | **ML Prediction Service** | Scores claims using XGBoost/LightGBM models to output `ml_risk_score` and `model_confidence`. |
| **6** | **Historical Similarity Service** | Generates claim embeddings and performs k-NN similarity searches against past claim outcomes. |
| **7** | **Risk / Rules Engine** | Evaluates configurable business rules (sanction lists, high-value thresholds, rapid submissions). |
| **8** | **Dynamic Decision Engine** | Combines ML risk, confidence, similarity, and rule signals into a final dynamic routing path. |
| **9** | **Claim Processing Service** | Executes the assigned route (Auto Process, Additional Validation, Human Investigation). |
| **10** | **Human Investigation API/UI** | Provides investigator workbench to review escalated claims, examine SHAP features, and submit verdicts. |
| **11** | **Feedback & Retraining Pipeline** | Persists investigator feedback and triggers model retraining cycles. |
| **12** | **Model Registry & Tracking** | Integrates MLflow for logging metrics, parameters, artifacts, and managing model versions. |
| **13** | **Monitoring & Observability** | Prometheus exporter exposing operational performance and ML metrics (drift, precision/recall). |

---

## 🛠️ Technology Stack

- **Core Language**: Python 3.10+
- **Data Engineering**: pandas, NumPy, scikit-learn
- **Machine Learning**: XGBoost, LightGBM, SHAP
- **Hyperparameter Tuning**: Optuna
- **MLOps & Tracking**: MLflow
- **Web API**: FastAPI, Uvicorn, Pydantic
- **Asynchronous Messaging**: RabbitMQ, aio-pika
- **Caching & Vector Search**: Redis, redis-py
- **Database & Persistence**: SQLite / PostgreSQL, SQLAlchemy 2.0
- **Observability**: Prometheus Client, Prometheus Server
- **Containerization**: Docker, Docker Compose
- **Frontend Workbench**: HTML5, Vanilla CSS (Glassmorphism), JavaScript (ES6+), Chart.js

---

## 📖 Execution & Workflow Specification

For an in-depth breakdown of execution flows, message schemas, CQRS handlers, MLOps retraining cycles, and decision matrices, please refer to [FLOW.md](file:///f:/Industry_project/FLOW.md).

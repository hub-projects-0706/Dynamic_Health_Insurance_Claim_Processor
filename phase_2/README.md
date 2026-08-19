# Healthcare Claim Model Training Pipeline (Phase 2)

An end-to-end Machine Learning training pipeline and interactive dashboard for healthcare claim fraud detection, built with **Python FastAPI**, **scikit-learn**, **React + Vite**, **PostgreSQL**, and **RabbitMQ**.

---

## 🚀 Key Features

- **Asynchronous ML Pipeline**: Submits training tasks to a **RabbitMQ** queue for decoupled background processing.
- **Multiple ML Algorithms**:
  - **Random Forest Classifier** (`RandomForestClassifier`)
  - **Gradient Tree Boosting** (`GradientBoostingClassifier`)
  - **Neural Network / Multilayer Perceptron** (`MLPClassifier` with `StandardScaler` feature normalization)
- **Clinical Data Ingestion**: Parses healthcare claims dataset (`dataset.csv`) or dynamically generates synthetic claims baselines. Extracts key ratios such as `costOverBenchmarkRatio` and `costVariance`.
- **Model Evaluation**: Computes real-time evaluation metrics: **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **Execution Time (ms)**.
- **Interactive UI Dashboard**: Built with **React** & **Recharts** to trigger ingestion, queue ML jobs, monitor job statuses, and visually compare model performance.

---

## 🛠 Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | React 18, Vite, Recharts, Lucide React, Vanilla CSS |
| **Backend API** | Python 3.11, FastAPI, Pydantic, Uvicorn |
| **Machine Learning** | scikit-learn, pandas, numpy |
| **Database** | PostgreSQL 15, SQLAlchemy ORM, psycopg2 |
| **Message Queue** | RabbitMQ 3 (AMQP protocol + Management Console) |
| **Containerization** | Docker, Docker Compose, Nginx |

---

## 📋 Data & Feature Specifications

The model trains on 8 clinical & financial interaction features extracted from claim records:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `claimedAmount` | Float | Total dollar amount billed on the claim |
| `regionalBenchmarkCost` | Float | Average benchmark cost for procedure in region |
| `codeMismatchScore` | Float | Anomaly score between procedure code & diagnosis |
| `priorClaimCount30d` | Int | Number of claims submitted by patient in last 30 days |
| `providerSanctionFlag` | Int (0/1) | Flag indicating if provider has past sanctions |
| `isDuplicateClaim` | Int (0/1) | Flag indicating potential duplicate submission |
| `costOverBenchmarkRatio` | Float | Computed feature: `claimedAmount / regionalBenchmarkCost` |
| `costVariance` | Float | Computed feature: `claimedAmount - regionalBenchmarkCost` |
| `isFraud` *(Target)* | Int (0/1) | Target classification label (0 = Legitimate, 1 = Fraud/Abuse) |

---

## 📁 Project Structure

```
phase_2/
├── backend/                  # Python FastAPI Backend
│   ├── data_ingestion.py     # CSV parsing & synthetic claim generator
│   ├── database.py           # SQLAlchemy database configuration
│   ├── Dockerfile            # Python 3.11 slim Docker container configuration
│   ├── main.py               # FastAPI application endpoints & lifecycle
│   ├── messaging.py          # RabbitMQ message publisher & background consumer
│   ├── ml_pipeline.py        # scikit-learn model training & metric evaluation
│   ├── models.py             # SQLAlchemy ORM models (ClaimData, TrainingJob, ModelMetric)
│   ├── requirements.txt      # Python dependencies
│   └── schemas.py            # Pydantic request & response DTO validation
├── frontend/                 # React + Vite Dashboard
│   ├── src/
│   │   ├── components/       # DataIngestionCard, TrainingControlCard, MetricsComparisonChart, JobsTable
│   │   ├── App.jsx           # Main Dashboard application component
│   │   └── main.jsx          # Entry point
│   ├── Dockerfile            # Multi-stage Nginx build for production static serving
│   ├── nginx.conf            # Nginx proxy configuration (/api -> backend:8080)
│   ├── package.json          # Node dependencies & scripts
│   └── vite.config.js        # Vite dev server configuration with API proxy
└── docker-compose.yml        # Orchestration for PostgreSQL, RabbitMQ, Backend, Frontend
```

---

## 🚦 Getting Started

### Method 1: Using Docker Compose (Recommended)

Run the entire multi-container stack with a single command from the root directory:

```powershell
docker-compose up --build
```

#### Access Points:
- 🌐 **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
- ⚙️ **FastAPI Interactive Docs (Swagger)**: [http://localhost:8080/docs](http://localhost:8080/docs)
- 🐇 **RabbitMQ Management Console**: [http://localhost:15672](http://localhost:15672) *(Credentials: `guest` / `guest`)*

---

### Method 2: Running Locally (Development Mode)

#### 1. Backend Setup (FastAPI)
```powershell
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start backend server
uvicorn main:app --reload --port 8080
```

#### 2. Frontend Setup (React + Vite)
```powershell
# Navigate to frontend directory
cd frontend

# Install node packages
npm install

# Start Vite dev server
npm run dev
```

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/claims/ingest` | Ingests raw claims dataset or generates synthetic baseline |
| `GET` | `/api/claims/count` | Returns total ingested claims count |
| `POST` | `/api/training/submit` | Enqueues model training job to RabbitMQ queue |
| `GET` | `/api/training/jobs` | Fetches list of training jobs & execution statuses |
| `GET` | `/api/training/jobs/{jobId}` | Gets status of a specific job |
| `GET` | `/api/metrics/compare` | Compares latest metrics across all algorithms |
| `GET` | `/api/metrics/history` | Fetches full historical model metrics log |

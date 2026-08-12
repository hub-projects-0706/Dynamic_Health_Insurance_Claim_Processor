## 📁 Repository Directory Structure

```text
.
├── README.md                           # Master project documentation
├── FLOW.md                             # Comprehensive execution & data flow specification
├── docker-compose.yml                  # Multi-container orchestration (FastAPI, RabbitMQ, Redis, Prometheus, MLflow)
├── Dockerfile                          # Application container definition
├── requirements.txt                    # Python dependencies
├── prometheus.yml                      # Prometheus scraper configuration
├── notebooks/
│   └── exploratory_analysis_and_modeling.ipynb  # EDA, model comparison, SHAP analysis
├── src/
│   ├── config.py                       # Application & environment configuration
│   ├── database/
│   │   ├── db.py                       # SQLAlchemy database initialization
│   │   ├── models.py                   # Data models (Claims, Events, Feedback, Models)
│   │   └── redis_client.py             # Redis caching & vector index interface
│   ├── messaging/
│   │   ├── rabbitmq.py                 # RabbitMQ event publisher & consumer
│   │   └── events.py                   # Event definitions & schemas
│   ├── cqrs/
│   │   ├── commands.py                 # Command definitions & write-model logic
│   │   ├── queries.py                  # Query definitions & read-model logic
│   │   └── handlers.py                 # Command and query handlers
│   ├── services/
│   │   ├── ingestion_service.py        # Ingestion & payload parsing
│   │   ├── validation_service.py       # Structural & domain validation
│   │   ├── feature_service.py          # Operational & behavioral feature engineering
│   │   ├── similarity_service.py       # Historical claim vector similarity matching
│   │   ├── ml_prediction_service.py    # XGBoost/LightGBM scoring & confidence estimation
│   │   ├── rules_engine.py             # Business rules evaluator
│   │   ├── decision_engine.py          # Dynamic routing matrix evaluator
│   │   ├── processing_service.py       # Route execution & state transition
│   │   └── retraining_service.py       # Optuna + MLflow automated retraining service
│   ├── mlops/
│   │   ├── trainer.py                  # Model trainer (XGBoost/LightGBM)
│   │   ├── tuning.py                   # Optuna hyperparameter tuning engine
│   │   ├── registry.py                 # MLflow tracking & model registry interface
│   │   └── drift_monitor.py            # Data & prediction drift calculator (PSI / KS-test)
│   ├── observability/
│   │   └── metrics.py                  # Prometheus metrics registry & exporters
│   └── api/
│       ├── main.py                     # FastAPI application entrypoint
│       ├── routes_claims.py            # CQRS Claims API routes
│       ├── routes_investigation.py     # Investigator queue & feedback API routes
│       ├── routes_mlops.py             # Model management & retraining API routes
│       └── routes_metrics.py           # Prometheus metrics endpoint
└── frontend/
    ├── index.html                      # Glassmorphic web application layout
    ├── css/
    │   └── styles.css                  # Modern UI design system & token definitions
    └── js/
        ├── app.js                      # Core application initialization & navigation
        ├── claims.js                   # Claim submission & live stream visualizer
        ├── investigator.js             # Investigator review workbench & feedback submission
        ├── mlops.js                    # MLflow experiment log & Optuna retrain trigger
        └── metrics.js                  # Prometheus system & ML observability charts
```

---
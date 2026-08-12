
## 🏗️ System Architecture

The system is designed as an **event-driven CQRS microservices platform**:

```text
                               Claim Submission
                                      │
                                      ▼
                                FastAPI Gateway
                                      │
                                  RabbitMQ
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
Validation Service            Data Pipeline                  Claim Service
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      ▼
                         Feature Engineering Service
                                      │
                                      ▼
                        ML Risk Prediction Service
                        (XGBoost / LightGBM + Confidence)
                                      │
                                      ▼
                    Historical Claim Intelligence Service
                    (Vector Similarity & Historical Outcomes)
                                      │
                                      ▼
                           Business Rules Engine
                                      │
                                      ▼
                           Dynamic Decision Engine
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
⚡ Auto Processing          🔍 Additional Validation     🕵️ Human Investigation
                                                                   │
                                                                   ▼
                                                            Human Feedback
                                                                   │
                                                                   ▼
                                                          Optuna & MLflow
                                                          Model Retraining
```
# 🤖 Task 4: End-to-End Machine Learning Baseline Model Pipeline

This folder contains the complete, reproducible **Task 4 Machine Learning Pipeline** for training a baseline model from raw data to serialized artifacts with evaluation metrics.

---

## 📁 Task 4 Artifacts & Deliverables Summary

| Artifact File | File Path | Type | Description |
|---|---|---|---|
| **Pipeline Script** | [`task-4/train_baseline_model.py`](file:///f:/Industry_project/ML-Project/task-4/train_baseline_model.py) | Python Script | Complete ML training, vectorization, evaluation, and artifact saving pipeline. |
| **Model Artifact** | [`task-4/baseline_model.joblib`](file:///f:/Industry_project/ML-Project/task-4/baseline_model.joblib) | Joblib Model | Trained XGBoost Classifier model binary (`random_state=42`). |
| **Preprocessor Artifact** | [`task-4/preprocessor.joblib`](file:///f:/Industry_project/ML-Project/task-4/preprocessor.joblib) | Joblib Pipeline | Fitted Scikit-Learn `ColumnTransformer` (StandardScaler + OneHotEncoder). |
| **Metrics Report** | [`task-4/evaluation_metrics.json`](file:///f:/Industry_project/ML-Project/task-4/evaluation_metrics.json) | JSON Metrics | Reproducible evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, CM). |
| **Documentation** | [`task-4/README.md`](file:///f:/Industry_project/ML-Project/task-4/README.md) | Documentation | Specification of model architecture, metrics breakdown, and execution steps. |

---

## 📐 Machine Learning Pipeline Architecture

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ 1. Dataset Ingestion    │ ──► │ 2. Preprocessor Vector  │ ──► │ 3. XGBoost Model        │ ──► │ 4. Metrics Evaluation   │
│    (`task-3/train.csv`) │     │    (`ColumnTransformer`)│     │    Training (Seed=42)   │     │    & Artifact Export    │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

1. **Dataset Ingestion**: Ingests prepared training data (`task-3/train_dataset.csv`, 1,180 rows) and testing evaluation data (`task-3/test_dataset.csv`, 200 rows).
2. **Scikit-Learn Vectorization Pipeline**: Fits a `ColumnTransformer` on training numerical features (`StandardScaler`) and categorical features (`OneHotEncoder(handle_unknown='ignore')`), producing a 573-dimensional feature vector.
3. **Model Training**: Fits an `XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)` baseline model.
4. **Evaluation & Persistence**: Transforms unseen testing samples (`test_dataset.csv`), evaluates classification metrics, saves `baseline_model.joblib`, `preprocessor.joblib`, and `evaluation_metrics.json`.

---

## 📊 Reproducible Evaluation Metrics Summary

Evaluated on 200 unseen testing samples from `task-3/test_dataset.csv`:

### Classification Performance Metrics Table

| Metric | Score Achieved | Percentage | Description / Significance |
|---|:---:|:---:|---|
| **Accuracy** | **0.7050** | **70.50%** | Proportion of overall correctly classified test claims. |
| **Precision** | **0.9083** | **90.83%** | High precision; minimal false accusations of fraud on legitimate claims. |
| **Recall** | **0.6689** | **66.89%** | Sensitivity in identifying fraudulent claims via baseline ML alone. |
| **F1-Score** | **0.7704** | — | Harmonic mean of Precision and Recall. |
| **ROC-AUC** | **0.8315** | — | Probability separation capability between clean and fraud claims. |

### Confusion Matrix Breakdown (200 Test Claims)

```text
                       Predicted Clean (0)    Predicted Fraud (1)
Actual Clean (0)              42 (TN)                10 (FP)
Actual Fraud (1)              49 (FN)                99 (TP)
```

* **True Negatives (TN)**: 42 legitimate claims correctly identified as clean.
* **False Positives (FP)**: 10 clean claims incorrectly flagged as suspicious.
* **False Negatives (FN)**: 49 fraudulent claims missed by the baseline model alone.
* **True Positives (TP)**: 99 fraudulent claims successfully detected.

---

## 🚀 How to Execute the Pipeline & Reproduce Results

To run the end-to-end baseline training pipeline from the project root (`ML-Project`):

```powershell
python task-4/train_baseline_model.py
```

### Programmatic Verification Snippet
```python
import joblib
import pandas as pd

model = joblib.load('task-4/baseline_model.joblib')
preprocessor = joblib.load('task-4/preprocessor.joblib')

test_df = pd.read_csv('task-3/test_dataset.csv')
features = [
    'claimed_amount', 'regional_benchmark_cost', 'code_mismatch_score',
    'prior_claim_count_30d', 'provider_sanction_flag', 'is_duplicate_claim',
    'cost_over_benchmark_ratio', 'cost_variance', 'is_emergency',
    'policy_status', 'icd10_diagnosis_code', 'cpt_procedure_code'
]

X_test_vec = preprocessor.transform(test_df[features])
predictions = model.predict(X_test_vec)

print("Predictions generated successfully for", len(predictions), "test samples.")
```

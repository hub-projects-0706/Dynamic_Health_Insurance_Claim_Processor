# 📊 Task 5: Baseline Model Performance Evaluation & Approach Documentation

This directory contains the complete **Task 5 deliverables**: metric calculation scripts, JSON evaluation reports, feature importance rankings, and documentation detailing the baseline machine learning approach for the **Adaptive Health Insurance Claim Intelligence & Dynamic Routing Platform**.

---

## 📁 Task 5 Artifacts & Deliverables Summary

| Deliverable Artifact | File Path | Type | Description |
|---|---|---|---|
| **Evaluation Script** | [`task-5/evaluate_baseline_metrics.py`](file:///f:/Industry_project/ML-Project/task-5/evaluate_baseline_metrics.py) | Python Script | Executable script computing comprehensive classification metrics and feature importances. |
| **Performance Metrics** | [`task-5/performance_metrics.json`](file:///f:/Industry_project/ML-Project/task-5/performance_metrics.json) | JSON Artifact | Complete empirical performance metrics for both unseen Test set (200 samples) and Train set (1,180 samples). |
| **Classification Report** | [`task-5/classification_report.json`](file:///f:/Industry_project/ML-Project/task-5/classification_report.json) | JSON Artifact | Per-class breakdown of Precision, Recall, F1-score, and Support for clean (`0`) and fraud (`1`) claims. |
| **Feature Importance** | [`task-5/feature_importance.json`](file:///f:/Industry_project/ML-Project/task-5/feature_importance.json) | JSON Artifact | Ranked feature importance list with relative weights and percentage contributions. |
| **Task Documentation** | [`task-5/README.md`](file:///f:/Industry_project/ML-Project/task-5/README.md) | Markdown | Full documentation of baseline approach, metric tables, overfit analysis, and execution steps. |

---

## 📐 Baseline Model Approach Architecture

The Task 4 baseline model evaluates health insurance claims via a supervised binary risk classifier trained on vectorized numerical and categorical features.

```
┌────────────────────────────────┐     ┌────────────────────────────────┐     ┌────────────────────────────────┐
│ Ingest Task 3 Datasets         │ ──► │ Feature Preprocessing Pipeline │ ──► │ Baseline Model Inference       │
│ • Train Set: 1,180 claims      │     │ • StandardScaler (Numerical)   │     │ • Model: XGBClassifier         │
│ • Test Set: 200 claims         │     │ • OneHotEncoder (Categorical)  │     │ • Config: depth=4, lr=0.05     │
└────────────────────────────────┘     └────────────────────────────────┘     └────────────────────────────────┘
                                                                                      │
                                                                                      ▼
┌────────────────────────────────┐     ┌────────────────────────────────┐     ┌────────────────────────────────┐
│ Export JSON Metric Reports     │ ◄── │ Feature Importance Ranking     │ ◄── │ Calculate Metric Suite         │
│ • performance_metrics.json     │     │ • Extract split-gain weights   │     │ • Accuracy, Precision, Recall  │
│ • classification_report.json   │     │ • Map to column names          │     │ • ROC-AUC, PR-AUC, Brier Score │
└────────────────────────────────┘     └────────────────────────────────┘     └────────────────────────────────┘
```

### Model Configuration Details
* **Algorithm**: `XGBClassifier` (Gradient Boosted Decision Trees)
* **Hyperparameters**: `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `random_state=42`, `eval_metric='logloss'`
* **Input Features**: 9 Numerical features (`claimed_amount`, `regional_benchmark_cost`, `code_mismatch_score`, `prior_claim_count_30d`, `provider_sanction_flag`, `is_duplicate_claim`, `cost_over_benchmark_ratio`, `cost_variance`, `is_emergency`) + 3 Categorical features (`policy_status`, `icd10_diagnosis_code`, `cpt_procedure_code`).
* **Vector Dimension**: 573 one-hot and scaled feature columns output by `preprocessor.joblib`.

---

## 📊 Comprehensive Performance Metrics Summary

Performance evaluated on 200 unseen testing samples from `task-3/test_dataset.csv`:

### 1. Overall Binary Classification Metrics

| Evaluation Metric | Score Achieved | Percentage | Metric Definition & Practical Interpretation |
|---|:---:|:---:|---|
| **Accuracy** | **0.7050** | **70.50%** | Proportion of overall correctly classified test claims (141 out of 200). |
| **Precision (Fraud)** | **0.9083** | **90.83%** | Proportion of predicted fraud claims that are actually fraudulent (99 TP out of 109 predicted fraud). Low false accusation rate. |
| **Recall / Sensitivity** | **0.6689** | **66.89%** | Proportion of actual fraudulent claims detected (99 TP out of 148 actual fraud claims). |
| **Specificity (TNR)** | **0.8077** | **80.77%** | Proportion of actual clean claims correctly identified (42 TN out of 52 clean claims). |
| **F1-Score** | **0.7704** | — | Harmonic mean of Precision and Recall. |
| **ROC-AUC Score** | **0.8315** | — | Area under the Receiver Operating Characteristic curve; strong probabilistic class separation capability. |
| **PR-AUC Score** | **0.9189** | — | Area under the Precision-Recall curve; evaluates performance on imbalanced positive class. |
| **Log Loss** | **0.5182** | — | Cross-entropy loss reflecting prediction probability uncertainty. |
| **Brier Score** | **0.1744** | — | Mean squared difference between predicted probability and actual outcome (calibration error). |

---

### 2. Confusion Matrix Breakdown (200 Test Claims)

```text
                       Predicted Clean (0)    Predicted Fraud (1)
Actual Clean (0)              42 (TN)                10 (FP)
Actual Fraud (1)              49 (FN)                99 (TP)
```

* **True Negatives (TN = 42)**: Clean, legitimate claims correctly classified for auto-processing.
* **False Positives (FP = 10)**: Clean claims falsely flagged as suspicious (False Alarm Rate: 19.23%).
* **False Negatives (FN = 49)**: Fraudulent/abusive claims missed by the ML model alone (Miss Rate: 33.11%).
* **True Positives (TP = 99)**: Fraudulent/abusive claims correctly identified by the baseline model.

---

### 3. Detailed Per-Class Classification Report

| Class Label | Class Description | Precision | Recall | F1-Score | Support (Samples) |
|:---:|---|:---:|:---:|:---:|:---:|
| **`0`** | Clean / Legitimate Claim | 0.4615 | 0.8077 | 0.5874 | 52 |
| **`1`** | Fraudulent / Suspicious Claim | 0.9083 | 0.6689 | 0.7704 | 148 |
| **Macro Avg** | Unweighted Average | 0.6849 | 0.7383 | 0.6789 | 200 |
| **Weighted Avg** | Sample-Weighted Average | 0.7921 | 0.7050 | 0.7228 | 200 |

---

## 📈 Generalization & Overfitting Diagnosis

Comparing train set performance (1,180 samples) against test set performance (200 samples) to evaluate generalization:

| Metric | Train Set (Reference) | Test Set (Unseen) | Generalization Gap ($\Delta$) | Status / Assessment |
|---|:---:|:---:|:---:|---|
| **Accuracy** | 82.54% | 70.50% | +12.04% | Mild Generalization Gap |
| **Precision** | 94.04% | 90.83% | +3.21% | Stable High Precision |
| **Recall** | 69.49% | 66.89% | +2.60% | Consistent Recall |
| **F1-Score** | 0.7992 | 0.7704 | +0.0288 | Excellent F1 Stability |
| **ROC-AUC** | 0.8979 | 0.8315 | +0.0664 | Moderate ROC Generalization |

### Key Diagnostic Takeaways
1. **High Precision Safety**: The baseline model exhibits **90.83% precision**, minimizing false accusations against legitimate policyholders.
2. **Detection Vulnerability (Recall = 66.89%)**: The baseline model misses 33.11% of fraudulent claims when operating as a standalone classifier at default decision threshold (0.50).
3. **Justification for Hybrid Architecture**: This performance profile validates the design of the platform's **Dynamic Decision Engine**, which supplements ML predictions with deterministic business rule penalties (`min(1.0, ML + Rule Penalty)`) to capture the remaining false negatives.

---

## 🔑 Top Feature Importance Analysis

The table below summarizes the top 10 feature drivers ranked by XGBoost split-gain importance:

| Rank | Feature Code / Name | Importance Weight | Contribution % | Domain Feature Description |
|:---:|---|:---:|:---:|---|
| **1** | `num__claimed_amount` | 0.1107 | **11.07%** | Requested claim reimbursement monetary amount. |
| **2** | `cat__policy_status_ACTIVE` | 0.1090 | **10.90%** | Active policy standing flag. |
| **3** | `num__provider_sanction_flag` | 0.0949 | **9.49%** | Medical license sanction / regulatory flag. |
| **4** | `num__is_duplicate_claim` | 0.0778 | **7.78%** | Multi-submission duplicate flag within 30 days. |
| **5** | `cat__icd10_diagnosis_code_72931` | 0.0743 | **7.43%** | Specific ICD-10 diagnosis code attribution. |
| **6** | `cat__icd10_diagnosis_code_1439` | 0.0580 | **5.80%** | Oncology/specialty diagnosis indicator. |
| **7** | `cat__policy_status_SUSPENDED` | 0.0429 | **4.29%** | Suspended member policy status indicator. |
| **8** | `cat__icd10_diagnosis_code_1972` | 0.0418 | **4.18%** | Clinical diagnosis code indicator. |
| **9** | `cat__policy_status_FRAUD_FLAGGED` | 0.0408 | **4.08%** | Known fraud historical flag. |
| **10** | `cat__icd10_diagnosis_code_78964` | 0.0351 | **3.51%** | Abdominal/symptom diagnosis code. |

---

## 🚀 How to Execute Evaluation & Reproduce Results

To run the Task 5 metric evaluation script and regenerate all JSON reports from the project root (`ML-Project`):

```powershell
python task-5/evaluate_baseline_metrics.py
```

### Verification Command
Inspect generated JSON files:
```powershell
Get-Content task-5/performance_metrics.json
```

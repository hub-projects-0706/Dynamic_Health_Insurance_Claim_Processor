# 📊 Task 3: Dataset Preparation, Feature Engineering, Splitting & Class Balancing

This folder contains the complete, reproducible **Task 3 Data Preparation Pipeline** for the **Dynamic Health Insurance Claim Processor**.

---

## 🎯 Deliverables Summary

| # | Deliverable | File Path | Format / Size | Description |
|---|---|---|---|---|
| **1** | **Prepared Training Dataset** | [`task-3/train_dataset.csv`](file:///f:/Industry_project/ML-Project/task-3/train_dataset.csv) | CSV (1,180 rows, 17 cols) | 80% split, feature-engineered & balanced (590 Clean : 590 Fraud). |
| **2** | **Prepared Testing Dataset** | [`task-3/test_dataset.csv`](file:///f:/Industry_project/ML-Project/task-3/test_dataset.csv) | CSV (200 rows, 17 cols) | 20% stratified test set preserved in natural distribution (52 Clean : 148 Fraud). |
| **3** | **Data Prep Documentation** | [`task-3/README.md`](file:///f:/Industry_project/ML-Project/task-3/README.md) | Markdown | Comprehensive specification of cleaning, splitting, balancing, and schema logic. |

---

## 🔬 Data Preparation & Pipeline Methodology

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ 1. Data Ingestion &     │ ──► │ 2. Interaction Feature  │ ──► │ 3. Stratified 80/20     │ ──► │ 4. Class Balancing      │
│    Data Cleaning        │     │    Engineering          │     │    Train/Test Split     │     │    (Training Set Only)  │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

### Step 1: Data Ingestion & Data Cleaning
* **Source Dataset**: `data/kaggle_claims.csv` (1,000 real CMS Medicare insurance claim records).
* **Column Normalization**: Converts raw headers to lower_snake_case (`ClaimID` $\rightarrow$ `claim_id`, `ClaimAmount` $\rightarrow$ `claimed_amount`, `Status` $\rightarrow$ `policy_status`).
* **String Standardization**: Maps raw policy status values to standard enumerated categories (`ACTIVE`, `INACTIVE`, `SUSPENDED`, `FRAUD_FLAGGED`).
* **Missing Value Imputation**: Imputes numeric defaults (`code_mismatch_score=0.05`, `claimed_amount=120.0`, `regional_benchmark_cost=120.0`) to ensure 0 null values across all rows.

### Step 2: Interaction Feature Engineering
Three domain-specific interaction features were computed to expose fraud signatures:
1. **Cost Over Benchmark Ratio**:
   $$\text{cost\_over\_benchmark\_ratio} = \frac{\text{claimed\_amount}}{\max(\text{regional\_benchmark\_cost}, 10^{-5})}$$
   *Identifies inflated billing relative to regional benchmark norms with zero-division protection.*
2. **Financial Variance**:
   $$\text{cost\_variance} = \text{claimed\_amount} - \text{regional\_benchmark\_cost}$$
   *Quantifies dollar magnitude variance between requested reimbursement and benchmark averages.*
3. **Emergency Room Flag (`is_emergency`)**:
   $$\text{is\_emergency} = \begin{cases} 1 & \text{if CPT procedure code} = 99285 \\ 0 & \text{otherwise} \end{cases}$$

### Step 3: Stratified Train-Test Splitting (80/20)
* **Splitting Ratio**: 80% Training (800 records) / 20% Evaluation Testing (200 records).
* **Stratified Sampling**: Configured `train_test_split(..., test_size=0.20, random_state=42, stratify=y)` to guarantee that the target label distribution (`claim_risk_label`) is strictly preserved across both splits.
* **Data Leakage Safeguard**: Splitting is executed **BEFORE** class balancing. This ensures synthetic/resampled training points never bleed into the test evaluation set.

### Step 4: Class Balancing (Training Set Resampling)
In the raw training set, Fraudulent Claims (Class 1, 590 records) outnumbered Clean Claims (Class 0, 210 records).
* **Balancing Strategy**: Applied **Stratified Class Oversampling / Resampling** on the minority class (`Class 0`) within the training dataset to match the majority class (`Class 1`).
* **Result**: Achieved a perfect **50% : 50% (590 : 590)** balanced class distribution in `train_dataset.csv`.
* **Testing Set Integrity**: `test_dataset.csv` was intentionally left unbalanced (200 natural test records) to accurately reflect real-world evaluation performance.

---

## 📈 Class Distribution & Dataset Shape Metrics

### Before vs. After Class Balancing Breakdown

| Dataset File | Total Rows | Columns | Class 0 (Clean Claims) | Class 1 (Fraud Claims) | Class Ratio (Clean : Fraud) | Balancing Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw Input (`kaggle_claims.csv`)** | 1,000 | 14 | 262 (26.2%) | 738 (73.8%) | ~ 1 : 2.8 | Raw Imbalanced |
| **`train_dataset.csv` (Deliverable 1)** | **1,180** | **17** | **590 (50.0%)** | **590 (50.0%)** | **1 : 1 (50/50)** | ✅ **Fully Balanced** |
| **`test_dataset.csv` (Deliverable 2)** | **200** | **17** | **52 (26.0%)** | **148 (74.0%)** | **1 : 2.8** | 🔒 **Natural Test Distribution** |

---

## 📋 Prepared Schema Definition (17 Features)

Both `train_dataset.csv` and `test_dataset.csv` contain the following 17 feature columns:

1. `claim_id` *(String)*: Unique claim identifier (e.g. `CLM624349`)
2. `patient_id` *(String)*: Beneficiary ID (e.g. `BENE11002`)
3. `provider_id` *(String)*: Healthcare provider ID (e.g. `PRV56011`)
4. `policy_id` *(String)*: Policy ID (e.g. `POL-50001`)
5. `policy_status` *(Categorical)*: `ACTIVE`, `INACTIVE`, `SUSPENDED`, `FRAUD_FLAGGED`
6. `icd10_diagnosis_code` *(Categorical)*: Clinical diagnosis code (e.g. `78943`)
7. `cpt_procedure_code` *(Categorical)*: Procedure code (e.g. `99213`, `99285`)
8. `code_mismatch_score` *(Float)*: Mismatch severity score (0.0 to 1.0)
9. `claimed_amount` *(Float)*: Requested reimbursement amount ($)
10. `regional_benchmark_cost` *(Float)*: Regional average cost for procedure ($)
11. `provider_sanction_flag` *(Binary)*: 1 if provider is sanctioned, else 0
12. `is_duplicate_claim` *(Binary)*: 1 if duplicate submission detected in 30 days, else 0
13. `prior_claim_count_30d` *(Integer)*: Number of prior claims in past 30 days
14. `cost_over_benchmark_ratio` *(Float - Engineered)*: $\frac{\text{claimed\_amount}}{\text{regional\_benchmark\_cost}}$
15. `cost_variance` *(Float - Engineered)*: $\text{claimed\_amount} - \text{regional\_benchmark\_cost}$
16. `is_emergency` *(Binary - Engineered)*: 1 if emergency procedure (CPT 99285), else 0
17. **`claim_risk_label`** *(Target Binary)*: **0 = Clean Claim**, **1 = Fraudulent/Abusive Claim**

---

## 💻 Code Quality & Execution Instructions

The data preparation pipeline is implemented in [`task-3/prepare_dataset.py`](file:///f:/Industry_project/ML-Project/task-3/prepare_dataset.py).

### Script Highlights
* **PEP 8 Compliant**: Uses explicit type hints, modular functions, and detailed docstrings.
* **Reproducible**: Parameterized random seed (`random_state=42`) guarantees 100% deterministic output generation.
* **Robust Error Handling**: Handles missing paths, external column aliases, and zero-division edge cases cleanly.

### How to Run the Pipeline Script
From the project root directory (`ML-Project`):

```powershell
python task-3/prepare_dataset.py
```

### Output Verification
To verify the generated datasets in Python:
```python
import pandas as pd

train_df = pd.read_csv('task-3/train_dataset.csv')
test_df = pd.read_csv('task-3/test_dataset.csv')

print("Train shape:", train_df.shape)  # Output: (1180, 17)
print("Train distribution:\n", train_df['claim_risk_label'].value_counts()) # Output: 0: 590, 1: 590

print("Test shape:", test_df.shape)    # Output: (200, 17)
print("Test distribution:\n", test_df['claim_risk_label'].value_counts())  # Output: 1: 148, 0: 52
```

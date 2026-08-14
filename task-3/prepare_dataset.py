"""
Task 3: Dataset Preparation, Feature Engineering, Stratified Splitting, and Class Balancing Pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK3_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

INPUT_DATA_PATH = os.path.join(DATA_DIR, 'kaggle_claims.csv')
TRAIN_OUTPUT_PATH = os.path.join(TASK3_DIR, 'train_dataset.csv')
TEST_OUTPUT_PATH = os.path.join(TASK3_DIR, 'test_dataset.csv')


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw dataset, standardizes column names, cleans missing values, and normalizes strings.
    """
    print(f"[STEP 1] Loading raw claims data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"   -> Loaded {len(df)} records with {df.shape[1]} columns.")

    # 1. Normalize Column Names
    df.columns = df.columns.str.strip().str.lower()

    # Alias handling if external dataset columns differ
    col_map = {
        'claimid': 'claim_id', 'policyid': 'policy_id', 'patientid': 'patient_id', 'providerid': 'provider_id',
        'status': 'policy_status', 'claimamount': 'claimed_amount', 'benchmarkcost': 'regional_benchmark_cost',
        'isfraud': 'claim_risk_label', 'potentialfraud': 'claim_risk_label'
    }
    df.rename(columns=col_map, inplace=True)

    # 2. String Field Normalization
    if 'policy_status' in df.columns:
        def clean_status(val):
            s = str(val).upper().strip()
            if 'FRAUD' in s or s == '1':
                return 'FRAUD_FLAGGED'
            elif 'SUSPEND' in s:
                return 'SUSPENDED'
            elif 'INACTIVE' in s or 'EXPIRED' in s:
                return 'INACTIVE'
            return 'ACTIVE'
        df['policy_status'] = df['policy_status'].apply(clean_status)

    # 3. Missing Value Imputation
    df['code_mismatch_score'] = df['code_mismatch_score'].fillna(0.05)
    df['claimed_amount'] = df['claimed_amount'].fillna(120.0)
    df['regional_benchmark_cost'] = df['regional_benchmark_cost'].fillna(120.0)
    df['provider_sanction_flag'] = df['provider_sanction_flag'].fillna(0).astype(int)
    df['is_duplicate_claim'] = df['is_duplicate_claim'].fillna(0).astype(int)
    df['prior_claim_count_30d'] = df['prior_claim_count_30d'].fillna(0).astype(int)

    print("   -> Data cleaning and normalization completed cleanly.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain interaction features: cost_over_benchmark_ratio, cost_variance, and emergency flags.
    """
    print("[STEP 2] Performing Feature Engineering...")
    df = df.copy()

    # 1. Cost Ratio over benchmark with safe division protection
    df['cost_over_benchmark_ratio'] = (
        df['claimed_amount'] / np.maximum(df['regional_benchmark_cost'], 1e-5)
    ).round(4)

    # 2. Financial Variance
    df['cost_variance'] = (df['claimed_amount'] - df['regional_benchmark_cost']).round(2)

    # 3. Emergency Flag Indicator (CPT 99285 = ER Visit)
    if 'cpt_procedure_code' in df.columns:
        df['is_emergency'] = (df['cpt_procedure_code'].astype(str) == '99285').astype(int)
    else:
        df['is_emergency'] = 0

    print(f"   -> Engineered 3 interaction features: cost_over_benchmark_ratio, cost_variance, is_emergency.")
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Splits dataset into 80% train and 20% test using Stratified Splitting to preserve label distribution.
    """
    print(f"[STEP 3] Splitting dataset into {(1-test_size)*100:.0f}% Train / {test_size*100:.0f}% Test (Stratified)...")
    
    X = df.drop(columns=['claim_risk_label'])
    y = df['claim_risk_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = X_train.copy()
    train_df['claim_risk_label'] = y_train

    test_df = X_test.copy()
    test_df['claim_risk_label'] = y_test

    print(f"   -> Raw Train set shape: {train_df.shape} | Risk label distribution: {dict(train_df['claim_risk_label'].value_counts())}")
    print(f"   -> Raw Test set shape:  {test_df.shape}  | Risk label distribution: {dict(test_df['claim_risk_label'].value_counts())}")

    return train_df, test_df


def balance_training_dataset(train_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Balances the training dataset using Stratified Resampling / Oversampling to achieve a balanced 50:50 class ratio.
    Data leakage prevention: Applied ONLY to the training set; the test set remains untouched.
    """
    print("[STEP 4] Balancing Training Dataset using Class Resampling...")
    
    df_majority = train_df[train_df['claim_risk_label'] == 1]
    df_minority = train_df[train_df['claim_risk_label'] == 0]

    count_maj = len(df_majority)
    count_min = len(df_minority)

    print(f"   -> Pre-balance Training Class Distribution: Class 1 (Fraud)={count_maj}, Class 0 (Clean)={count_min}")

    # Oversample minority class to match majority class
    df_minority_oversampled = resample(
        df_minority,
        replace=True,
        n_samples=count_maj,
        random_state=random_state
    )

    train_balanced = pd.concat([df_majority, df_minority_oversampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"   -> Post-balance Training Class Distribution: Class 1={sum(train_balanced['claim_risk_label']==1)}, Class 0={sum(train_balanced['claim_risk_label']==0)}")
    print(f"   -> Total Balanced Training Samples: {len(train_balanced)}")

    return train_balanced


def run_pipeline():
    """
    Executes end-to-end dataset preparation task.
    """
    print("==================================================================")
    print("      TASK 3: DATASET PREPARATION & BALANCING PIPELINE           ")
    print("==================================================================")

    # Step 1: Load and Clean
    df_clean = load_and_clean_data(INPUT_DATA_PATH)

    # Step 2: Feature Engineering
    df_engineered = engineer_features(df_clean)

    # Step 3: Stratified Train-Test Split
    raw_train, test_df = split_dataset(df_engineered, test_size=0.20, random_state=42)

    # Step 4: Class Balancing on Training Set Only
    train_balanced = balance_training_dataset(raw_train, random_state=42)

    # Step 5: Export Deliverables
    print("[STEP 5] Exporting CSV Deliverables...")
    train_balanced.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print(f"   [CREATED] Prepared Training Dataset: {TRAIN_OUTPUT_PATH} ({len(train_balanced)} rows)")
    print(f"   [CREATED] Prepared Testing Dataset:  {TEST_OUTPUT_PATH} ({len(test_df)} rows)")
    print("==================================================================")
    print("   TASK 3 DATASET PREPARATION COMPLETED SUCCESSFULLY!            ")
    print("==================================================================")


if __name__ == '__main__':
    run_pipeline()

"""
Task 4: End-to-End Machine Learning Pipeline - Raw Data to Baseline Model with Reproducible Evaluation Metrics.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier

# Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK4_DIR = os.path.dirname(os.path.abspath(__file__))
TASK3_DIR = os.path.join(PROJECT_ROOT, 'task-3')

TRAIN_DATA_PATH = os.path.join(TASK3_DIR, 'train_dataset.csv')
TEST_DATA_PATH = os.path.join(TASK3_DIR, 'test_dataset.csv')
MODEL_OUTPUT_PATH = os.path.join(TASK4_DIR, 'baseline_model.joblib')
PREPROCESSOR_OUTPUT_PATH = os.path.join(TASK4_DIR, 'preprocessor.joblib')
METRICS_OUTPUT_PATH = os.path.join(TASK4_DIR, 'evaluation_metrics.json')

NUMERICAL_FEATURES = [
    'claimed_amount',
    'regional_benchmark_cost',
    'code_mismatch_score',
    'prior_claim_count_30d',
    'provider_sanction_flag',
    'is_duplicate_claim',
    'cost_over_benchmark_ratio',
    'cost_variance',
    'is_emergency'
]

CATEGORICAL_FEATURES = [
    'policy_status',
    'icd10_diagnosis_code',
    'cpt_procedure_code'
]


def load_datasets():
    """
    Loads prepared training and testing datasets from Task 3.
    """
    print(f"[STEP 1] Ingesting training dataset from: {TRAIN_DATA_PATH}")
    print(f"[STEP 1] Ingesting testing dataset from:  {TEST_DATA_PATH}")

    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError("Task 3 datasets not found. Please run task-3/prepare_dataset.py first.")

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    print(f"   -> Loaded Train Set: {train_df.shape[0]} rows, {train_df.shape[1]} columns.")
    print(f"   -> Loaded Test Set:  {test_df.shape[0]} rows, {test_df.shape[1]} columns.")

    return train_df, test_df


def build_and_fit_preprocessor(train_df: pd.DataFrame):
    """
    Builds and fits a ColumnTransformer on numerical and categorical features.
    """
    print("[STEP 2] Building Scikit-Learn Feature Transformation Pipeline...")

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    X_train_raw = train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df['claim_risk_label'].values

    X_train_transformed = preprocessor.fit_transform(X_train_raw)

    print(f"   -> Preprocessing complete. Transformed Feature Vector Dimension: {X_train_transformed.shape[1]}")
    return preprocessor, X_train_transformed, y_train


def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """
    Trains an XGBoost Baseline Classifier with a deterministic random seed.
    """
    print("[STEP 3] Training XGBoost Classifier Baseline Model...")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    print("   -> XGBoost Baseline Model training completed successfully.")
    return model


def evaluate_model(model: XGBClassifier, preprocessor: ColumnTransformer, test_df: pd.DataFrame) -> dict:
    """
    Evaluates baseline model performance on unseen testing dataset.
    """
    print("[STEP 4] Evaluating Model Performance on Unseen Evaluation Test Set...")

    X_test_raw = test_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df['claim_risk_label'].values

    X_test_transformed = preprocessor.transform(X_test_raw)

    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'model_type': 'XGBClassifier',
        'random_seed': 42,
        'test_samples_count': int(len(y_test)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': {
            'true_negatives_clean': int(tn),
            'false_positives_false_fraud': int(fp),
            'false_negatives_missed_fraud': int(fn),
            'true_positives_caught_fraud': int(tp)
        }
    }

    print("\n==================================================================")
    print("   REPRODUCIBLE BASELINE EVALUATION METRICS METRICS              ")
    print("==================================================================")
    print(f"   • Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   • Precision:      {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   • Recall:         {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   • F1-Score:       {metrics['f1_score']:.4f}")
    print(f"   • ROC-AUC Score:  {metrics['roc_auc']:.4f}")
    print("\n   Confusion Matrix Breakdown:")
    print(f"     - True Negatives  (Clean correctly identified): {tn}")
    print(f"     - False Positives (Clean accused of fraud):    {fp}")
    print(f"     - False Negatives (Fraud missed):              {fn}")
    print(f"     - True Positives  (Fraud caught):              {tp}")
    print("==================================================================\n")

    return metrics


def save_artifacts(model: XGBClassifier, preprocessor: ColumnTransformer, metrics: dict):
    """
    Saves model, preprocessor, and metrics artifacts to disk.
    """
    print("[STEP 5] Saving Model & Preprocessor Artifacts...")

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"   [SAVED] Trained Model Artifact:       {MODEL_OUTPUT_PATH}")

    joblib.dump(preprocessor, PREPROCESSOR_OUTPUT_PATH)
    print(f"   [SAVED] Feature Preprocessor Artifact: {PREPROCESSOR_OUTPUT_PATH}")

    with open(METRICS_OUTPUT_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   [SAVED] Evaluation Metrics JSON:      {METRICS_OUTPUT_PATH}")


def run_pipeline():
    """
    Executes end-to-end Task 4 machine learning training pipeline.
    """
    print("==================================================================")
    print("      TASK 4: ML PIPELINE & BASELINE TRAINING ENGINE             ")
    print("==================================================================")

    # 1. Load Data
    train_df, test_df = load_datasets()

    # 2. Build & Fit Preprocessor
    preprocessor, X_train, y_train = build_and_fit_preprocessor(train_df)

    # 3. Train Model
    model = train_baseline_model(X_train, y_train)

    # 4. Evaluate Model
    metrics = evaluate_model(model, preprocessor, test_df)

    # 5. Save Artifacts
    save_artifacts(model, preprocessor, metrics)

    print("==================================================================")
    print("   TASK 4 BASELINE MODEL PIPELINE COMPLETED SUCCESSFULLY!        ")
    print("==================================================================")


if __name__ == '__main__':
    run_pipeline()

"""
Task 5: Baseline Model Performance Evaluation & Feature Importance Computation
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK5_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_DIR = os.path.join(PROJECT_ROOT, 'task-4')
TASK3_DIR = os.path.join(PROJECT_ROOT, 'task-3')

TRAIN_DATA_PATH = os.path.join(TASK3_DIR, 'train_dataset.csv')
TEST_DATA_PATH = os.path.join(TASK3_DIR, 'test_dataset.csv')
MODEL_PATH = os.path.join(TASK4_DIR, 'baseline_model.joblib')
PREPROCESSOR_PATH = os.path.join(TASK4_DIR, 'preprocessor.joblib')

METRICS_JSON_PATH = os.path.join(TASK5_DIR, 'performance_metrics.json')
CLASS_REPORT_JSON_PATH = os.path.join(TASK5_DIR, 'classification_report.json')
FEATURE_IMPORTANCE_JSON_PATH = os.path.join(TASK5_DIR, 'feature_importance.json')

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


def load_artifacts_and_data():
    """
    Loads baseline model, preprocessor, and prepared train/test datasets.
    """
    print(f"[TASK 5 STEP 1] Loading model artifact: {MODEL_PATH}")
    print(f"[TASK 5 STEP 1] Loading preprocessor artifact: {PREPROCESSOR_PATH}")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Task 4 model artifacts not found. Run task-4/train_baseline_model.py first.")
        
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    print(f"   -> Train Dataset: {len(train_df)} rows")
    print(f"   -> Test Dataset:  {len(test_df)} rows")
    
    return model, preprocessor, train_df, test_df


def compute_metrics(model, preprocessor, df, set_name="Test"):
    """
    Computes comprehensive binary classification performance metrics.
    """
    X_raw = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_true = df['claim_risk_label'].values
    
    X_vec = preprocessor.transform(X_raw)
    
    y_pred = model.predict(X_vec)
    y_proba = model.predict_proba(X_vec)[:, 1]
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    loss = float(log_loss(y_true, y_proba))
    brier = float(brier_score_loss(y_true, y_proba))
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        'set_name': set_name,
        'sample_count': int(len(y_true)),
        'accuracy': acc,
        'precision': prec,
        'recall_sensitivity': rec,
        'specificity_tnr': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'log_loss': loss,
        'brier_score': brier,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'confusion_matrix': {
            'true_negatives_clean': int(tn),
            'false_positives_false_fraud': int(fp),
            'false_negatives_missed_fraud': int(fn),
            'true_positives_caught_fraud': int(tp)
        }
    }
    
    return metrics, report_dict, X_vec, y_true, y_pred, y_proba


def compute_feature_importances(model, preprocessor):
    """
    Extracts feature importances and maps them to human-readable encoded names.
    """
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    
    total_importance = sum(importances) if sum(importances) > 0 else 1.0
    
    fi_list = []
    for name, imp in zip(feature_names, importances):
        fi_list.append({
            'feature': name,
            'importance_weight': float(imp),
            'percentage_contribution': float((imp / total_importance) * 100)
        })
        
    fi_sorted = sorted(fi_list, key=lambda x: x['importance_weight'], reverse=True)
    return fi_sorted


def main():
    print("==================================================================")
    print("      TASK 5: BASELINE MODEL PERFORMANCE EVALUATION              ")
    print("==================================================================")
    
    # 1. Load Artifacts
    model, preprocessor, train_df, test_df = load_artifacts_and_data()
    
    # 2. Evaluate Performance
    test_metrics, test_report, X_test_vec, y_test, y_pred_test, y_proba_test = compute_metrics(
        model, preprocessor, test_df, set_name="Test Set (Unseen)"
    )
    train_metrics, train_report, _, _, _, _ = compute_metrics(
        model, preprocessor, train_df, set_name="Train Set (Reference)"
    )
    
    # 3. Compute Feature Importances
    feature_importances = compute_feature_importances(model, preprocessor)
    
    combined_performance = {
        'task': 'Task 5 Baseline Performance Metric Evaluation',
        'model_name': 'XGBClassifier Baseline',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'random_state': 42
        },
        'test_set_evaluation': test_metrics,
        'train_set_evaluation': train_metrics,
        'overfit_gap': {
            'accuracy_delta': float(train_metrics['accuracy'] - test_metrics['accuracy']),
            'f1_delta': float(train_metrics['f1_score'] - test_metrics['f1_score']),
            'roc_auc_delta': float(train_metrics['roc_auc'] - test_metrics['roc_auc'])
        }
    }
    
    # 4. Save JSON Artifacts
    print("\n[TASK 5 STEP 2] Saving performance metrics JSON artifacts...")
    
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump(combined_performance, f, indent=2)
    print(f"   [SAVED] Performance Metrics:       {METRICS_JSON_PATH}")
    
    with open(CLASS_REPORT_JSON_PATH, 'w') as f:
        json.dump({'test_set_classification_report': test_report}, f, indent=2)
    print(f"   [SAVED] Classification Report:     {CLASS_REPORT_JSON_PATH}")
    
    with open(FEATURE_IMPORTANCE_JSON_PATH, 'w') as f:
        json.dump({'top_features': feature_importances[:25]}, f, indent=2)
    print(f"   [SAVED] Feature Importance List:   {FEATURE_IMPORTANCE_JSON_PATH}")
    
    # Print Summary Table
    print("\n==================================================================")
    print("               TASK 5 PERFORMANCE METRICS SUMMARY                ")
    print("==================================================================")
    print(f"   • Accuracy:               {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"   • Precision (Fraud):      {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"   • Recall / Sensitivity:   {test_metrics['recall_sensitivity']:.4f} ({test_metrics['recall_sensitivity']*100:.2f}%)")
    print(f"   • Specificity (Clean TNR): {test_metrics['specificity_tnr']:.4f} ({test_metrics['specificity_tnr']*100:.2f}%)")
    print(f"   • F1-Score:               {test_metrics['f1_score']:.4f}")
    print(f"   • ROC-AUC Score:          {test_metrics['roc_auc']:.4f}")
    print(f"   • PR-AUC Score:           {test_metrics['pr_auc']:.4f}")
    print(f"   • Log Loss:               {test_metrics['log_loss']:.4f}")
    print(f"   • Brier Score:            {test_metrics['brier_score']:.4f}")
    print("\n   Confusion Matrix (200 Test Samples):")
    print(f"     - True Negatives  (TN): {test_metrics['confusion_matrix']['true_negatives_clean']}")
    print(f"     - False Positives (FP): {test_metrics['confusion_matrix']['false_positives_false_fraud']}")
    print(f"     - False Negatives (FN): {test_metrics['confusion_matrix']['false_negatives_missed_fraud']}")
    print(f"     - True Positives  (TP): {test_metrics['confusion_matrix']['true_positives_caught_fraud']}")
    print("\n   Top 5 Predictive Features:")
    for idx, fi in enumerate(feature_importances[:5], 1):
        print(f"     {idx}. {fi['feature']}: {fi['percentage_contribution']:.2f}%")
    print("==================================================================\n")


if __name__ == '__main__':
    main()

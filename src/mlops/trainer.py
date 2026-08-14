import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier
from src.services.feature_service import preprocess_data

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model.joblib')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.joblib')


def train_model(data_path: str = 'data/dataset.csv') -> dict:
    """
    Executes model training for Dynamic Health Insurance Claim Processor.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[INFO] Loading Health Insurance Claims dataset from {data_path}...")
    df = pd.read_csv(data_path)

    print("[INFO] Preprocessing & engineering claim interaction features...")
    X, y, preprocessor = preprocess_data(df, is_training=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Training XGBoost Claim Classifier on {X_train.shape[0]} claim records...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    print("\n[EVAL] Dynamic Health Insurance Claim Processor - Evaluation Metrics:")
    print(f"   - ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"   - F1-Score:   {metrics['f1_score']:.4f}")
    print(f"   - Precision:  {metrics['precision']:.4f}")
    print(f"   - Recall:     {metrics['recall']:.4f}")
    print(f"   - Accuracy:   {metrics['accuracy']:.4f}\n")

    print(f"[SAVE] Saving trained model artifact to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    print(f"[SAVE] Saving preprocessor pipeline artifact to {PREPROCESSOR_PATH}...")
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("[SUCCESS] Health Insurance Claim model training and artifact persistence completed successfully!")
    return metrics


if __name__ == '__main__':
    train_model()

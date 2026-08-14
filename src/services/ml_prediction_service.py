import os
import joblib
import pandas as pd
import numpy as np
from src.services.feature_service import preprocess_data

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model.joblib')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.joblib')

_model = None
_preprocessor = None


def _load_artifacts():
    global _model, _preprocessor
    if _model is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(
                f"Model or preprocessor artifacts not found at {MODEL_DIR}. "
                "Please run 'python src/train.py' first."
            )
        _model = joblib.load(MODEL_PATH)
        _preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict_risk(payload: dict) -> dict:
    """
    Evaluates ML risk score and model confidence for a single input record payload.
    """
    _load_artifacts()

    # Convert dictionary payload to single-row DataFrame
    df_raw = pd.DataFrame([payload])

    # Transform features using pre-fitted preprocessor
    X_transformed, _, _ = preprocess_data(df_raw, preprocessor=_preprocessor, is_training=False)

    # Predict risk probability score
    probabilities = _model.predict_proba(X_transformed)[0]
    risk_score = float(probabilities[1])

    # Compute classification confidence metric (0.0 to 1.0)
    confidence = float(abs(risk_score - 0.5) * 2.0)

    # Predict binary decision flag
    prediction_label = int(risk_score >= 0.5)

    return {
        'risk_score': round(risk_score, 4),
        'confidence': round(confidence, 4),
        'prediction_label': prediction_label
    }

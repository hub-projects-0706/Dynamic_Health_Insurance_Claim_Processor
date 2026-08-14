import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.services.feature_service import engineer_raw_features, preprocess_data

def test_feature_engineering_ratios():
    raw_df = pd.DataFrame([{
        'claimed_amount': 240.0,
        'regional_benchmark_cost': 120.0,
        'policy_status': 'ACTIVE',
        'icd10_diagnosis_code': 'J06.9',
        'cpt_procedure_code': '99213'
    }])

    engineered = engineer_raw_features(raw_df)

    assert 'cost_over_benchmark_ratio' in engineered.columns
    assert 'cost_variance' in engineered.columns
    assert pytest.approx(engineered['cost_over_benchmark_ratio'].iloc[0], 0.01) == 2.0
    assert pytest.approx(engineered['cost_variance'].iloc[0], 0.01) == 120.0

def test_external_dataset_normalization():
    external_df = pd.DataFrame([{
        'ClaimID': 'CLM-99',
        'PolicyID': 'POL-88',
        'Status': 'FRAUD_FLAGGED',
        'ClaimAmount': 500.0,
        'BenchmarkCost': 250.0
    }])

    engineered = engineer_raw_features(external_df)

    assert engineered['policy_status'].iloc[0] == 'FRAUD_FLAGGED'
    assert engineered['claimed_amount'].iloc[0] == 500.0
    assert engineered['regional_benchmark_cost'].iloc[0] == 250.0

def test_preprocessor_transformation_shape():
    raw_df = pd.DataFrame([{
        'claimed_amount': 150.0,
        'regional_benchmark_cost': 120.0,
        'code_mismatch_score': 0.1,
        'prior_claim_count_30d': 1,
        'provider_sanction_flag': 0,
        'is_duplicate_claim': 0,
        'policy_status': 'ACTIVE',
        'icd10_diagnosis_code': 'J06.9',
        'cpt_procedure_code': '99213',
        'claim_risk_label': 0
    }])

    X, y, preprocessor = preprocess_data(raw_df, is_training=True)

    assert X.shape[0] == 1
    assert X.shape[1] > 0
    assert y[0] == 0

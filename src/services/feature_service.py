import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERICAL_FEATURES = [
    'claimed_amount',
    'regional_benchmark_cost',
    'code_mismatch_score',
    'prior_claim_count_30d',
    'provider_sanction_flag',
    'is_duplicate_claim',
    'cost_over_benchmark_ratio',
    'cost_variance'
]

CATEGORICAL_FEATURES = [
    'policy_status',
    'icd10_diagnosis_code',
    'cpt_procedure_code'
]


def normalize_external_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes external datasets (e.g. Kaggle, CMS, CSV uploads) into the standard schema.
    """
    df = df.copy()

    # Column Mapping Aliases
    col_mappings = {
        'ClaimID': 'claim_id', 'ClaimNo': 'claim_id',
        'PolicyID': 'policy_id', 'PolicyNo': 'policy_id',
        'PatientID': 'patient_id', 'BeneID': 'patient_id',
        'ProviderID': 'provider_id', 'Provider': 'provider_id',
        'PolicyStatus': 'policy_status', 'Status': 'policy_status',
        'DiagnosisCode': 'icd10_diagnosis_code', 'ICD10': 'icd10_diagnosis_code', 'ClmDiagnosisCode_1': 'icd10_diagnosis_code',
        'ProcedureCode': 'cpt_procedure_code', 'CPT': 'cpt_procedure_code', 'ClmProcedureCode_1': 'cpt_procedure_code',
        'ClaimAmount': 'claimed_amount', 'InscClaimAmtReimbursed': 'claimed_amount', 'Amount': 'claimed_amount',
        'BenchmarkCost': 'regional_benchmark_cost', 'Benchmark': 'regional_benchmark_cost',
        'PotentialFraud': 'claim_risk_label', 'IsFraud': 'claim_risk_label', 'fraud': 'claim_risk_label', 'churn': 'claim_risk_label'
    }

    df.rename(columns=col_mappings, inplace=True)

    # Defaults for missing columns
    if 'claim_id' not in df.columns:
        df['claim_id'] = [f'CLM-{10001 + i}' for i in range(len(df))]
    if 'policy_id' not in df.columns:
        df['policy_id'] = [f'POL-{50001 + i}' for i in range(len(df))]
    if 'patient_id' not in df.columns:
        df['patient_id'] = 'PAT-8000'
    if 'provider_id' not in df.columns:
        df['provider_id'] = 'PRV-100'

    if 'policy_status' not in df.columns:
        df['policy_status'] = 'ACTIVE'
    else:
        # Normalize Policy Status string values to uppercase standard (ACTIVE, INACTIVE, SUSPENDED, FRAUD_FLAGGED)
        def map_status(val):
            s = str(val).upper().strip()
            if 'FRAUD' in s or s == '1' or s == 'YES':
                return 'FRAUD_FLAGGED'
            elif 'SUSPEND' in s:
                return 'SUSPENDED'
            elif 'INACTIVE' in s or 'EXPIRED' in s or 'LAPSE' in s:
                return 'INACTIVE'
            else:
                return 'ACTIVE'
        df['policy_status'] = df['policy_status'].apply(map_status)

    if 'icd10_diagnosis_code' not in df.columns:
        df['icd10_diagnosis_code'] = 'J06.9'
    if 'cpt_procedure_code' not in df.columns:
        df['cpt_procedure_code'] = '99213'
    if 'code_mismatch_score' not in df.columns:
        df['code_mismatch_score'] = 0.05
    if 'claimed_amount' not in df.columns:
        df['claimed_amount'] = 120.0
    if 'regional_benchmark_cost' not in df.columns:
        df['regional_benchmark_cost'] = 120.0
    if 'provider_sanction_flag' not in df.columns:
        df['provider_sanction_flag'] = 0
    if 'is_duplicate_claim' not in df.columns:
        df['is_duplicate_claim'] = 0
    if 'prior_claim_count_30d' not in df.columns:
        df['prior_claim_count_30d'] = 1

    return df


def engineer_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers health insurance claim interaction terms & financial variance metrics.
    """
    df = normalize_external_dataset(df)

    # Derived interaction & ratio metrics
    df['cost_over_benchmark_ratio'] = df['claimed_amount'] / np.maximum(df['regional_benchmark_cost'], 1e-5)
    df['cost_variance'] = df['claimed_amount'] - df['regional_benchmark_cost']

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Builds an unfitted ColumnTransformer pipeline for health insurance claim features.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def preprocess_data(df: pd.DataFrame, preprocessor: ColumnTransformer = None, is_training: bool = True):
    """
    Applies health claim feature engineering and ColumnTransformer pipeline.
    """
    df_engineered = engineer_raw_features(df)

    # Target variable extraction
    target_col = 'claim_risk_label' if 'claim_risk_label' in df_engineered.columns else None
    y = df_engineered[target_col].values if target_col is not None else None

    if is_training:
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(df_engineered)
    else:
        if preprocessor is None:
            raise ValueError("Fitted preprocessor must be provided when is_training=False")
        X_transformed = preprocessor.transform(df_engineered)

    return X_transformed, y, preprocessor

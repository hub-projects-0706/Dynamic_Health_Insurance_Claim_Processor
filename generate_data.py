import numpy as np
import pandas as pd
import os

os.makedirs('data', exist_ok=True)
np.random.seed(42)
n_samples = 600

claim_ids = [f'CLM-{10001 + i}' for i in range(n_samples)]
policy_ids = [f'POL-{50001 + np.random.randint(0, 400)}' for _ in range(n_samples)]
patient_ids = [f'PAT-{8000 + np.random.randint(0, 300)}' for _ in range(n_samples)]
provider_ids = [f'PRV-{101 + np.random.randint(0, 50)}' for _ in range(n_samples)]

policy_statuses = np.random.choice(
    ['ACTIVE', 'INACTIVE', 'SUSPENDED', 'FRAUD_FLAGGED'],
    size=n_samples,
    p=[0.75, 0.12, 0.08, 0.05]
)

icd10_codes = np.random.choice(['J06.9', 'I10', 'E11.9', 'M54.5', 'S82.0'], size=n_samples, p=[0.30, 0.25, 0.20, 0.15, 0.10])

# CPT mapping base benchmark amounts
cpt_benchmark_map = {
    '99213': 120.0,   # Low/Mod Office Visit
    '99214': 185.0,   # Complex Office Visit
    '72148': 850.0,   # MRI Lumbar Spine
    '27447': 8500.0,  # Total Knee Replacement
    '99285': 1400.0   # ER Visit High Severity
}

cpt_codes = np.random.choice(list(cpt_benchmark_map.keys()), size=n_samples, p=[0.35, 0.25, 0.18, 0.10, 0.12])
regional_benchmarks = np.array([cpt_benchmark_map[code] for code in cpt_codes])

# Cost variance generation
cost_multipliers = np.random.choice([0.9, 1.0, 1.1, 1.5, 2.2, 3.0], size=n_samples, p=[0.40, 0.35, 0.10, 0.08, 0.04, 0.03])
claimed_amounts = np.round(regional_benchmarks * cost_multipliers + np.random.normal(0, 15, size=n_samples), 2)
claimed_amounts = np.maximum(claimed_amounts, 50.0)

# Clinical code mismatch score (0.00 to 1.00)
code_mismatch_scores = np.round(np.random.beta(a=1.5, b=5.0, size=n_samples), 4)

# Provider sanction flag & duplicate flags
provider_sanction_flags = np.random.choice([0, 1], size=n_samples, p=[0.94, 0.06])
is_duplicate_claims = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
prior_claim_count_30d = np.random.poisson(lam=1.2, size=n_samples)

# Risk Label Generation Logic
log_odds = (
    -2.2
    + 3.5 * (policy_statuses == 'FRAUD_FLAGGED')
    + 2.0 * (policy_statuses == 'SUSPENDED')
    + 1.2 * (policy_statuses == 'INACTIVE')
    + 2.2 * provider_sanction_flags
    + 2.5 * is_duplicate_claims
    + 2.8 * (code_mismatch_scores > 0.65)
    + 1.5 * (claimed_amounts / regional_benchmarks > 1.6)
    + 0.4 * (prior_claim_count_30d >= 4)
)
prob = 1 / (1 + np.exp(-log_odds))
claim_risk_labels = (np.random.rand(n_samples) < prob).astype(int)

df = pd.DataFrame({
    'claim_id': claim_ids,
    'policy_id': policy_ids,
    'patient_id': patient_ids,
    'provider_id': provider_ids,
    'policy_status': policy_statuses,
    'icd10_diagnosis_code': icd10_codes,
    'cpt_procedure_code': cpt_codes,
    'code_mismatch_score': code_mismatch_scores,
    'claimed_amount': claimed_amounts,
    'regional_benchmark_cost': regional_benchmarks,
    'provider_sanction_flag': provider_sanction_flags,
    'is_duplicate_claim': is_duplicate_claims,
    'prior_claim_count_30d': prior_claim_count_30d,
    'claim_risk_label': claim_risk_labels
})

df.to_csv('data/dataset.csv', index=False)
fraud_rate = df['claim_risk_label'].mean()
print(f"[SUCCESS] Updated dataset.csv with policy_id & 4 Policy Statuses (ACTIVE, INACTIVE, SUSPENDED, FRAUD_FLAGGED). Fraud rate: {fraud_rate:.2%}")

def evaluate_rules(payload: dict) -> dict:
    """
    Evaluates Health Insurance Claim Adjudication rules & Policy ID / Status compliance flags.
    """
    policy_id = payload.get('policy_id', 'UNKNOWN_POLICY')
    policy_status = payload.get('policy_status', 'ACTIVE')
    provider_sanction = payload.get('provider_sanction_flag', 0)
    is_duplicate = payload.get('is_duplicate_claim', 0)
    mismatch_score = payload.get('code_mismatch_score', 0.0)
    claimed_amt = payload.get('claimed_amount', 0.0)
    benchmark_amt = payload.get('regional_benchmark_cost', 1.0)
    prior_claims = payload.get('prior_claim_count_30d', 0)

    cost_ratio = claimed_amt / max(benchmark_amt, 1e-5)

    triggered_rules = []
    rule_risk_penalty = 0.0
    critical_flags = []

    # 1. Policy ID & Policy Status Audit
    if policy_status == 'FRAUD_FLAGGED':
        flag = f"FLAG_POLICY_BLACK_LISTED_FRAUD [{policy_id}]"
        triggered_rules.append(flag)
        critical_flags.append(flag)
        rule_risk_penalty += 0.40
    elif policy_status == 'SUSPENDED':
        flag = f"FLAG_POLICY_ADMINISTRATIVELY_SUSPENDED [{policy_id}]"
        triggered_rules.append(flag)
        critical_flags.append(flag)
        rule_risk_penalty += 0.35
    elif policy_status == 'INACTIVE':
        flag = f"FLAG_POLICY_INACTIVE_LAPSED [{policy_id}]"
        triggered_rules.append(flag)
        critical_flags.append(flag)
        rule_risk_penalty += 0.30

    # 2. Provider Sanction Audit
    if provider_sanction == 1:
        flag = "FLAG_PROVIDER_SANCTIONED"
        triggered_rules.append(flag)
        critical_flags.append(flag)
        rule_risk_penalty += 0.30

    # 3. Duplicate Claim Audit
    if is_duplicate == 1:
        flag = "FLAG_DUPLICATE_CLAIM_SUBMISSION"
        triggered_rules.append(flag)
        critical_flags.append(flag)
        rule_risk_penalty += 0.35

    # 4. Clinical Code Mismatch Audit (ICD-10 vs CPT)
    if mismatch_score >= 0.65:
        flag = "FLAG_HIGH_CLINICAL_CODE_MISMATCH"
        triggered_rules.append(flag)
        critical_flags.append(flag)
        rule_risk_penalty += 0.25

    # 5. Financial Cost Benchmark Audit
    if cost_ratio >= 1.60:
        flag = "FLAG_EXCESSIVE_COST_OVER_BENCHMARK"
        triggered_rules.append(flag)
        rule_risk_penalty += 0.20

    # 6. Submission Velocity Audit
    if prior_claims >= 4:
        flag = "FLAG_RAPID_CLAIM_FREQUENCY_30D"
        triggered_rules.append(flag)
        rule_risk_penalty += 0.15

    has_critical_flag = len(critical_flags) > 0

    return {
        'policy_id': policy_id,
        'policy_status': policy_status,
        'triggered_rules': triggered_rules,
        'critical_flags': critical_flags,
        'rule_risk_penalty': round(rule_risk_penalty, 4),
        'has_critical_flag': has_critical_flag
    }

from src.services.rules_engine import evaluate_rules

def test_clean_active_policy():
    payload = {
        'policy_id': 'POL-100',
        'policy_status': 'ACTIVE',
        'provider_sanction_flag': 0,
        'is_duplicate_claim': 0,
        'code_mismatch_score': 0.05,
        'claimed_amount': 120.0,
        'regional_benchmark_cost': 120.0,
        'prior_claim_count_30d': 1
    }

    result = evaluate_rules(payload)

    assert result['has_critical_flag'] is False
    assert len(result['triggered_rules']) == 0
    assert result['rule_risk_penalty'] == 0.0

def test_blacklisted_fraud_policy():
    payload = {
        'policy_id': 'POL-999',
        'policy_status': 'FRAUD_FLAGGED',
        'provider_sanction_flag': 1,
        'is_duplicate_claim': 1,
        'code_mismatch_score': 0.85,
        'claimed_amount': 15000.0,
        'regional_benchmark_cost': 8500.0,
        'prior_claim_count_30d': 5
    }

    result = evaluate_rules(payload)

    assert result['has_critical_flag'] is True
    assert any('BLACK_LISTED_FRAUD' in r for r in result['triggered_rules'])
    assert 'FLAG_PROVIDER_SANCTIONED' in result['triggered_rules']
    assert 'FLAG_DUPLICATE_CLAIM_SUBMISSION' in result['triggered_rules']
    assert result['rule_risk_penalty'] > 0.60

import sys
import os

# Ensure src root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.services.ml_prediction_service import predict_risk
from src.services.rules_engine import evaluate_rules


def evaluate_claim(payload: dict) -> dict:
    """
    Evaluates multi-signal intelligence (ML claim risk prediction + Policy Status & Health Insurance Audit Rules)
    and applies the Dynamic Decision Routing Matrix.
    """
    ml_res = predict_risk(payload)
    rule_res = evaluate_rules(payload)

    risk_score = ml_res['risk_score']
    confidence = ml_res['confidence']
    penalty = rule_res['rule_risk_penalty']
    critical_flag = rule_res['has_critical_flag']
    policy_status = rule_res['policy_status']
    cpt_code = payload.get('cpt_procedure_code', '')
    is_emergency = payload.get('is_emergency', 0) or (cpt_code == '99285')

    # Composite risk score capped at 1.0
    composite_risk_score = round(min(1.0, risk_score + penalty), 4)

    # 1. FRAUD / BLACKLISTED / CRITICAL RISK ESCALATION
    if composite_risk_score > 0.65 or critical_flag or policy_status in ['SUSPENDED', 'FRAUD_FLAGGED']:
        route = 'HUMAN_INVESTIGATION'
        priority = 'HIGH_PRIORITY_FRAUD_AUDIT'
        reason = f"Policy & Compliance Alert [{policy_status}]: Escalated to Human Investigator due to high risk score or critical compliance audit violations."

    # 2. EMERGENCY / CRITICAL FAST-TRACK AUTO-APPROVAL
    elif is_emergency and composite_risk_score < 0.35 and not critical_flag and policy_status == 'ACTIVE':
        route = 'AUTO_PROCESSED'
        priority = 'EMERGENCY_INSTANT_FAST_TRACK'
        reason = "[FAST-TRACK] Emergency Fast-Track: Instant AI approval granted in <10ms to avoid treatment delays."

    # 3. ROUTINE AUTO-APPROVAL
    elif composite_risk_score < 0.35 and confidence >= 0.20 and not critical_flag and policy_status == 'ACTIVE':
        route = 'AUTO_PROCESSED'
        priority = 'STANDARD_AUTO_APPROVAL'
        reason = "Routine AI Fast-Track: Low risk score, active policy, and high model confidence with clean compliance audit."

    # 4. BORDERLINE / SECONDARY VALIDATION
    else:
        route = 'PENDING_ADDITIONAL_VALIDATION'
        priority = 'SECONDARY_VALIDATION_QUEUE'
        reason = "Moderate risk score or inactive policy requiring secondary automated validation."

    return {
        'claim_id': payload.get('claim_id', 'UNKNOWN'),
        'policy_id': payload.get('policy_id', 'UNKNOWN_POLICY'),
        'policy_status': policy_status,
        'patient_id': payload.get('patient_id', 'UNKNOWN'),
        'provider_id': payload.get('provider_id', 'UNKNOWN'),
        'route': route,
        'processing_priority': priority,
        'composite_risk_score': composite_risk_score,
        'ml_risk_score': risk_score,
        'model_confidence': confidence,
        'triggered_rules': rule_res['triggered_rules'],
        'critical_flags': rule_res['critical_flags'],
        'routing_reason': reason
    }


def test_decision_engine():
    """
    Test suite demonstrating Policy Status & Dynamic Routing.
    """
    sample_payloads = [
        {
            'claim_id': 'CLM-30001',
            'policy_id': 'POL-98741',
            'patient_id': 'PAT-7001',
            'provider_id': 'PRV-110',
            'policy_status': 'ACTIVE',
            'icd10_diagnosis_code': 'S82.0',
            'cpt_procedure_code': '99285',
            'is_emergency': 1,
            'code_mismatch_score': 0.02,
            'claimed_amount': 1400.00,
            'regional_benchmark_cost': 1400.00,
            'provider_sanction_flag': 0,
            'is_duplicate_claim': 0,
            'prior_claim_count_30d': 0
        },
        {
            'claim_id': 'CLM-30002',
            'policy_id': 'POL-99999',
            'patient_id': 'PAT-7002',
            'provider_id': 'PRV-199',
            'policy_status': 'FRAUD_FLAGGED',
            'icd10_diagnosis_code': 'J06.9',
            'cpt_procedure_code': '27447',
            'is_emergency': 0,
            'code_mismatch_score': 0.92,
            'claimed_amount': 18500.00,
            'regional_benchmark_cost': 8500.00,
            'provider_sanction_flag': 1,
            'is_duplicate_claim': 1,
            'prior_claim_count_30d': 6
        }
    ]

    print("=========================================================")
    print("   Testing Policy ID & Status Dynamic Routing           ")
    print("=========================================================")

    for i, p in enumerate(sample_payloads, 1):
        decision = evaluate_claim(p)
        print(f"\n[CLAIM {i}] Claim ID: {decision['claim_id']} | Policy ID: {decision['policy_id']} | Status: {decision['policy_status']}")
        print(f"   - Dynamic Route:        {decision['route']}")
        print(f"   - Composite Risk Score: {decision['composite_risk_score']}")
        print(f"   - Triggered Rules:      {decision['triggered_rules']}")
        print(f"   - Reason:               {decision['routing_reason']}")


if __name__ == '__main__':
    test_decision_engine()

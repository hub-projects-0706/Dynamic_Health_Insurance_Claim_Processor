import json
import sys
import os

# Ensure src root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.services.decision_engine import evaluate_claim

def run_json_evaluations(json_file_path="data/sample_claims.json"):
    print("=========================================================")
    print("  Evaluating Test Claim Records with Policy IDs & Status ")
    print("=========================================================\n")

    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    test_claims = data.get("test_claims", [])

    for item in test_claims:
        test_name = item.get("test_name", "Test Claim")
        payload = item.get("payload", {})

        decision = evaluate_claim(payload)

        print(f"[TEST] {test_name}")
        print(f"   - Claim ID:             {decision['claim_id']}")
        print(f"   - Policy ID:            {decision['policy_id']}")
        print(f"   - Policy Status:        {decision['policy_status']}")
        print(f"   - Dynamic Route:        {decision['route']}")
        print(f"   - Composite Risk Score: {decision['composite_risk_score']}")
        print(f"   - ML Risk Score:        {decision['ml_risk_score']}")
        print(f"   - Triggered Audit Rules:{decision['triggered_rules']}")
        print(f"   - Routing Reason:       {decision['routing_reason']}\n" + "-"*55 + "\n")

if __name__ == '__main__':
    run_json_evaluations()

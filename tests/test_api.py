from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Dynamic Health Insurance Claim Processor" in data["service"]

def test_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "roc_auc" in data
    assert data["accuracy"] > 0.70

def test_samples_endpoint():
    response = client.get("/api/v1/claims/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 3

def test_evaluate_claim_endpoint():
    payload = {
        "claim_id": "CLM-TEST-01",
        "policy_id": "POL-50112",
        "patient_id": "PAT-8012",
        "provider_id": "PRV-105",
        "policy_status": "ACTIVE",
        "icd10_diagnosis_code": "J06.9",
        "cpt_procedure_code": "99213",
        "code_mismatch_score": 0.05,
        "claimed_amount": 125.00,
        "regional_benchmark_cost": 120.00,
        "provider_sanction_flag": 0,
        "is_duplicate_claim": 0,
        "prior_claim_count_30d": 1
    }

    response = client.post("/api/v1/claims/evaluate", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "route" in res
    assert "composite_risk_score" in res
    assert res["route"] in ["AUTO_PROCESSED", "PENDING_ADDITIONAL_VALIDATION", "HUMAN_INVESTIGATION"]

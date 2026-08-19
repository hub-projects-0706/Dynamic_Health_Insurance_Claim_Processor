import os
import csv
import logging
from sqlalchemy.orm import Session
from models import ClaimData

logger = logging.getLogger("claims_backend")


def ingest_default_dataset(db: Session) -> int:
    logger.info("Ingesting default healthcare claims dataset...")
    claims = []

    # Possible candidate paths for dataset.csv
    candidate_paths = [
        os.path.join("..", "data", "dataset.csv"),
        os.path.join("data", "dataset.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv"),
        os.path.join(os.path.dirname(__file__), "data", "dataset.csv"),
    ]

    csv_file_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_file_path = p
            break

    if csv_file_path:
        try:
            with open(csv_file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 14:
                        claim_id = row[0].strip()
                        code_mismatch = parse_float(row[7], 0.1)
                        claimed_amount = parse_float(row[8], 150.0)
                        benchmark_cost = parse_float(row[9], 120.0)
                        sanction_flag = parse_int(row[10], 0)
                        duplicate_flag = parse_int(row[11], 0)
                        prior_claims = parse_int(row[12], 0)
                        is_fraud = parse_int(row[13], 0)

                        cost_ratio = claimed_amount / benchmark_cost if benchmark_cost > 0 else 1.0
                        cost_variance = claimed_amount - benchmark_cost

                        claim = ClaimData(
                            claim_id=claim_id,
                            claimed_amount=claimed_amount,
                            regional_benchmark_cost=benchmark_cost,
                            code_mismatch_score=code_mismatch,
                            prior_claim_count30d=prior_claims,
                            provider_sanction_flag=sanction_flag,
                            is_duplicate_claim=duplicate_flag,
                            cost_over_benchmark_ratio=cost_ratio,
                            cost_variance=cost_variance,
                            is_fraud=is_fraud
                        )
                        claims.append(claim)
            logger.info("Parsed %d claims from CSV file: %s", len(claims), csv_file_path)
        except Exception as e:
            logger.warning("Failed to parse CSV file, generating synthetic dataset fallback: %s", e)

    # Fallback synthetic dataset generator if claims list is empty
    if not claims:
        logger.info("Generating synthetic claims dataset for ML pipeline...")
        for i in range(1, 201):
            is_fraud_case = (i % 4 == 0)
            claimed = 8500.0 + (i * 15.0) if is_fraud_case else 150.0 + (i * 2.0)
            benchmark = 120.0
            mismatch = 0.75 if is_fraud_case else 0.05
            sanctions = 1 if is_fraud_case else 0
            duplicates = 1 if is_fraud_case else 0
            prior = 5 if is_fraud_case else 1
            label = 1 if is_fraud_case else 0

            ratio = claimed / benchmark
            variance = claimed - benchmark

            claim = ClaimData(
                claim_id=f"CLM-SYN-{i}",
                claimed_amount=claimed,
                regional_benchmark_cost=benchmark,
                code_mismatch_score=mismatch,
                prior_claim_count30d=prior,
                provider_sanction_flag=sanctions,
                is_duplicate_claim=duplicates,
                cost_over_benchmark_ratio=ratio,
                cost_variance=variance,
                is_fraud=label
            )
            claims.append(claim)

    # Clear previous claims and bulk save new claims
    db.query(ClaimData).delete()
    db.add_all(claims)
    db.commit()
    logger.info("Successfully persisted %d claims into database.", len(claims))
    return len(claims)


def parse_float(val: str, default_val: float) -> float:
    try:
        return float(val.strip())
    except Exception:
        return default_val


def parse_int(val: str, default_val: int) -> int:
    try:
        return int(val.strip())
    except Exception:
        return default_val

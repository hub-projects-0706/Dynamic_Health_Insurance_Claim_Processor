import time
import logging
import random
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import ClaimData, TrainingJob, ModelMetric, JobStatus, AlgorithmType
from database import SessionLocal
import data_ingestion

logger = logging.getLogger("claims_backend")


def execute_training_job(job_id: str, algorithm_type_str: str, db: Session = None):
    close_session_on_exit = False
    if db is None:
        db = SessionLocal()
        close_session_on_exit = True

    try:
        logger.info("Executing training job [%s] with algorithm [%s]", job_id, algorithm_type_str)

        # Normalize algorithm_type enum string
        try:
            algo_enum = AlgorithmType(algorithm_type_str)
        except ValueError:
            algo_enum = AlgorithmType.RANDOM_FOREST

        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            job = TrainingJob(job_id=job_id, algorithm_type=algo_enum.value, status=JobStatus.QUEUED.value)
            db.add(job)
            db.commit()

        job.status = JobStatus.IN_PROGRESS.value
        db.commit()

        start_time = time.time()

        # Check claim count and auto-ingest if baseline data is missing
        claim_count = db.query(ClaimData).count()
        if claim_count == 0:
            logger.info("No claims found in database. Auto-ingesting dataset...")
            data_ingestion.ingest_default_dataset(db)

        all_claims = db.query(ClaimData).all()
        if not all_claims:
            raise RuntimeError("No claim data available for model training.")

        # Shuffle claims & split 80% train / 20% test
        random.seed(42)
        claims_list = list(all_claims)
        random.shuffle(claims_list)

        split_idx = int(len(claims_list) * 0.8)
        train_claims = claims_list[:split_idx]
        test_claims = claims_list[split_idx:]

        # Extract features (X) and labels (y)
        X_train, y_train = extract_features_and_labels(train_claims)
        X_test, y_test = extract_features_and_labels(test_claims)

        # Build and fit algorithm model
        if algo_enum == AlgorithmType.RANDOM_FOREST:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        elif algo_enum == AlgorithmType.GRADIENT_BOOSTING:
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        elif algo_enum == AlgorithmType.NEURAL_NETWORK:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=200, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type_str}")

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Calculate evaluation metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        # Save metrics record
        metric = ModelMetric(
            job_id=job_id,
            algorithm_type=algo_enum.value,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            execution_time_ms=execution_time_ms,
            sample_count=len(test_claims),
            timestamp=datetime.utcnow()
        )
        db.add(metric)

        # Update TrainingJob status
        job.status = JobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        db.commit()

        logger.info("Training job [%s] completed successfully in %d ms! Accuracy=%.4f", job_id, execution_time_ms, acc)

    except Exception as e:
        logger.error("Error executing training job [%s]: %s", job_id, e, exc_info=True)
        if job:
            job.status = JobStatus.FAILED.value
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        if close_session_on_exit:
            db.close()


def extract_features_and_labels(claims):
    features = []
    labels = []
    for c in claims:
        row = [
            c.claimed_amount,
            c.regional_benchmark_cost,
            c.code_mismatch_score,
            c.prior_claim_count30d,
            c.provider_sanction_flag,
            c.is_duplicate_claim,
            c.cost_over_benchmark_ratio,
            c.cost_variance
        ]
        features.append(row)
        labels.append(c.is_fraud)
    return np.array(features), np.array(labels)

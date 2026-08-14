import sys
import os

# Ensure src root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, RawClaim, ProcessedClaim, ProcessingLog, RejectedClaim

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'claims_platform.db')
DATABASE_URL = os.environ.get('DATABASE_URL', f'sqlite:///{DB_PATH}')

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    Initializes database tables according to schema.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)
    print(f"[SUCCESS] Database initialized successfully at {DATABASE_URL}")


def get_db():
    """
    Dependency generator for FastAPI DB session management.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_processed_claim_record(db, claim_payload: dict, evaluation_result: dict):
    """
    Persists raw claim, processed claim decision, and operational log into the database.
    """
    try:
        claim_id = claim_payload.get('claim_id')

        # Check existing processed claim
        existing = db.query(ProcessedClaim).filter(ProcessedClaim.claim_id == claim_id).first()
        if not existing:
            processed_rec = ProcessedClaim(
                claim_id=claim_id,
                policy_id=claim_payload.get('policy_id', 'UNKNOWN'),
                route=evaluation_result.get('route', 'UNKNOWN'),
                processing_priority=evaluation_result.get('processing_priority', 'NORMAL'),
                composite_risk_score=evaluation_result.get('composite_risk_score', 0.0),
                ml_risk_score=evaluation_result.get('ml_risk_score', 0.0),
                model_confidence=evaluation_result.get('model_confidence', 0.0),
                routing_reason=evaluation_result.get('routing_reason', '')
            )
            db.add(processed_rec)

            # Log operational trace
            log_rec = ProcessingLog(
                trace_id=f"TRACE-{claim_id}",
                claim_id=claim_id,
                status=evaluation_result.get('route'),
                message=f"Claim processed: Route={evaluation_result.get('route')}, Risk={evaluation_result.get('composite_risk_score')}"
            )
            db.add(log_rec)

            # If rejected/critical fraud, log rejected claim
            if evaluation_result.get('route') == 'HUMAN_INVESTIGATION':
                reject_rec = RejectedClaim(
                    claim_id=claim_id,
                    policy_id=claim_payload.get('policy_id', 'UNKNOWN'),
                    rejection_reason=evaluation_result.get('routing_reason', ''),
                    critical_flags=",".join(evaluation_result.get('critical_flags', []))
                )
                db.add(reject_rec)

            db.commit()
    except Exception as e:
        db.rollback()
        print(f"[WARN] DB persistence warning: {e}")


if __name__ == '__main__':
    init_db()

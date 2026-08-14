from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class RawClaim(Base):
    __tablename__ = "raw_claims"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String(50), unique=True, nullable=False, index=True)
    policy_id = Column(String(50), nullable=False)
    patient_id = Column(String(50), nullable=False)
    provider_id = Column(String(50), nullable=False)
    policy_status = Column(String(30), default="ACTIVE")
    icd10_diagnosis_code = Column(String(30), nullable=False)
    cpt_procedure_code = Column(String(30), nullable=False)
    code_mismatch_score = Column(Float, default=0.0)
    claimed_amount = Column(Float, nullable=False)
    regional_benchmark_cost = Column(Float, nullable=False)
    provider_sanction_flag = Column(Integer, default=0)
    is_duplicate_claim = Column(Integer, default=0)
    prior_claim_count_30d = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ProcessedClaim(Base):
    __tablename__ = "processed_claims"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String(50), unique=True, nullable=False, index=True)
    policy_id = Column(String(50), nullable=False)
    route = Column(String(40), nullable=False)
    processing_priority = Column(String(50), nullable=False)
    composite_risk_score = Column(Float, nullable=False)
    ml_risk_score = Column(Float, nullable=False)
    model_confidence = Column(Float, nullable=False)
    routing_reason = Column(Text)
    processed_at = Column(DateTime, default=datetime.utcnow)


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(100), nullable=False)
    claim_id = Column(String(50))
    status = Column(String(30), nullable=False)
    message = Column(Text, nullable=False)
    logged_at = Column(DateTime, default=datetime.utcnow)


class RejectedClaim(Base):
    __tablename__ = "rejected_claims"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String(50), nullable=False)
    policy_id = Column(String(50), nullable=False)
    rejection_reason = Column(Text, nullable=False)
    critical_flags = Column(Text)
    rejected_at = Column(DateTime, default=datetime.utcnow)

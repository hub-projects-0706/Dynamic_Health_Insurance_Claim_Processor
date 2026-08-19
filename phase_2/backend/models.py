import enum
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Enum as SQLEnum, BigInteger
from database import Base


class AlgorithmType(str, enum.Enum):
    RANDOM_FOREST = "RANDOM_FOREST"
    GRADIENT_BOOSTING = "GRADIENT_BOOSTING"
    NEURAL_NETWORK = "NEURAL_NETWORK"


class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ClaimData(Base):
    __tablename__ = "claims_data"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    claim_id = Column(String, index=True)
    claimed_amount = Column(Float)
    regional_benchmark_cost = Column(Float)
    code_mismatch_score = Column(Float)
    prior_claim_count30d = Column(Integer)
    provider_sanction_flag = Column(Integer)
    is_duplicate_claim = Column(Integer)
    cost_over_benchmark_ratio = Column(Float)
    cost_variance = Column(Float)
    is_fraud = Column(Integer)  # Target label (0 = Legitimate, 1 = Fraud/Abuse)


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    job_id = Column(String, primary_key=True, index=True)
    algorithm_type = Column(String)  # RANDOM_FOREST, GRADIENT_BOOSTING, NEURAL_NETWORK
    status = Column(String, default=JobStatus.QUEUED.value)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(String, index=True)
    algorithm_type = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    execution_time_ms = Column(BigInteger)
    sample_count = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

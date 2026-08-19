import uuid
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import engine, Base, get_db
from models import ClaimData, TrainingJob, ModelMetric, AlgorithmType, JobStatus
from schemas import (
    TrainingRequestDto,
    JobStatusResponseDto,
    MetricsComparisonDto,
    IngestResponse,
    ClaimCountResponse,
)
import data_ingestion
import messaging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("claims_backend")

# Create database tables if they do not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Healthcare Claim ML Training Pipeline",
    description="Python FastAPI backend replacing Java Spring Boot pipeline",
    version="1.0.0"
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    logger.info("Initializing Healthcare Claim ML Backend Service...")
    # Start background RabbitMQ consumer daemon thread
    messaging.start_rabbitmq_consumer()


@app.get("/")
def read_root():
    return {"message": "Healthcare Claim Model Training Pipeline API is running."}


# ──────────────────────────────────────────
# Claims Endpoints
# ──────────────────────────────────────────
@app.post("/api/claims/ingest", response_model=IngestResponse)
def ingest_claims(db: Session = Depends(get_db)):
    count = data_ingestion.ingest_default_dataset(db)
    return {
        "status": "SUCCESS",
        "message": "Ingested healthcare claims dataset.",
        "ingestedCount": count
    }


@app.get("/api/claims/count", response_model=ClaimCountResponse)
def get_claim_count(db: Session = Depends(get_db)):
    count = db.query(ClaimData).count()
    return {"totalClaims": count}


# ──────────────────────────────────────────
# Training Control Endpoints
# ──────────────────────────────────────────
@app.post("/api/training/submit", response_model=JobStatusResponseDto)
def submit_training_job(request: TrainingRequestDto, db: Session = Depends(get_db)):
    algo_type = request.algorithmType if request.algorithmType else AlgorithmType.RANDOM_FOREST
    job_id = "JOB-" + uuid.uuid4().hex[:8].upper()

    job = TrainingJob(
        job_id=job_id,
        algorithm_type=algo_type.value,
        status=JobStatus.QUEUED.value,
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Publish to RabbitMQ / Async Worker Queue
    messaging.publish_training_job(job_id, algo_type.value)

    return map_job_to_dto(job)


@app.get("/api/training/jobs", response_model=List[JobStatusResponseDto])
def get_all_jobs(db: Session = Depends(get_db)):
    jobs = db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).all()
    return [map_job_to_dto(j) for j in jobs]


@app.get("/api/training/jobs/{job_id}", response_model=JobStatusResponseDto)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return map_job_to_dto(job)


# ──────────────────────────────────────────
# Metrics Endpoints
# ──────────────────────────────────────────
@app.get("/api/metrics/compare", response_model=List[MetricsComparisonDto])
def get_comparison_metrics(db: Session = Depends(get_db)):
    result = []
    for algo in [AlgorithmType.RANDOM_FOREST, AlgorithmType.GRADIENT_BOOSTING, AlgorithmType.NEURAL_NETWORK]:
        latest_metric = (
            db.query(ModelMetric)
            .filter(ModelMetric.algorithm_type == algo.value)
            .order_by(ModelMetric.timestamp.desc())
            .first()
        )
        if latest_metric:
            result.append(map_metric_to_dto(latest_metric))
    return result


@app.get("/api/metrics/history", response_model=List[MetricsComparisonDto])
def get_metrics_history(db: Session = Depends(get_db)):
    metrics = db.query(ModelMetric).order_by(ModelMetric.timestamp.desc()).all()
    return [map_metric_to_dto(m) for m in metrics]


def map_job_to_dto(job: TrainingJob) -> JobStatusResponseDto:
    return JobStatusResponseDto(
        jobId=job.job_id,
        algorithmType=job.algorithm_type,
        status=job.status,
        errorMessage=job.error_message,
        createdAt=job.created_at,
        completedAt=job.completed_at
    )


def map_metric_to_dto(m: ModelMetric) -> MetricsComparisonDto:
    return MetricsComparisonDto(
        jobId=m.job_id,
        algorithmType=m.algorithm_type,
        accuracy=m.accuracy,
        precision=m.precision,
        recall=m.recall,
        f1Score=m.f1_score,
        executionTimeMs=m.execution_time_ms,
        sampleCount=m.sample_count
    )

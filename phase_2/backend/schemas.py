from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from models import AlgorithmType, JobStatus


class TrainingRequestDto(BaseModel):
    algorithmType: Optional[AlgorithmType] = Field(default=AlgorithmType.RANDOM_FOREST)


class JobStatusResponseDto(BaseModel):
    jobId: str
    algorithmType: str
    status: str
    errorMessage: Optional[str] = None
    createdAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None

    class Config:
        from_attributes = True


class MetricsComparisonDto(BaseModel):
    jobId: str
    algorithmType: str
    accuracy: float
    precision: float
    recall: float
    f1Score: float
    executionTimeMs: int
    sampleCount: int

    class Config:
        from_attributes = True


class IngestResponse(BaseModel):
    status: str
    message: str
    ingestedCount: int


class ClaimCountResponse(BaseModel):
    totalClaims: int

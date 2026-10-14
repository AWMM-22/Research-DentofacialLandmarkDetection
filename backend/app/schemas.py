"""Pydantic request/response schemas for the FastAPI service."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AnalysisStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class FrameQuality(str, Enum):
    accepted = "accepted"
    rejected = "rejected"
    flagged = "flagged"   # usable but below confidence threshold


class LandmarkSource(str, Enum):
    model = "model"
    manual = "manual"


class PointModel(BaseModel):
    x: float
    y: float


class LandmarkPoint(BaseModel):
    x: float
    y: float
    confidence: Optional[float] = None
    source: LandmarkSource = LandmarkSource.model


class LandmarksModel(BaseModel):
    nasion: Optional[LandmarkPoint] = None
    menton: Optional[LandmarkPoint] = None
    zygion_right: Optional[LandmarkPoint] = None
    zygion_left: Optional[LandmarkPoint] = None


class FrameMeasurements(BaseModel):
    height: float
    width: float
    facial_index: float


class FrameRejection(BaseModel):
    reason: str
    detail: Optional[str] = None


class FrameResult(BaseModel):
    frame_id: str
    frame_index: int
    timestamp_ms: int
    status: FrameQuality
    landmarks: LandmarksModel
    measurements: Optional[FrameMeasurements] = None
    rejection: Optional[FrameRejection] = None
    blur_variance: Optional[float] = None
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    face_box: Optional[tuple[float, float, float, float]] = None  # x,y,w,h


class QualitySummary(BaseModel):
    sampled: int = 0
    accepted: int = 0
    rejected: int = 0
    flagged: int = 0
    rejection_reasons: dict[str, int] = Field(default_factory=dict)


class AggregateResult(BaseModel):
    facial_index: Optional[float] = None
    category: str
    median_height: Optional[float] = None
    median_width: Optional[float] = None
    accepted_frame_count: int = 0


class ModelInfo(BaseModel):
    name: str
    version: str
    note: str


class AnalysisResult(BaseModel):
    job_id: str
    status: AnalysisStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    filename: Optional[str] = None
    model: ModelInfo
    quality: QualitySummary
    aggregate: AggregateResult
    representative_frame_id: Optional[str] = None
    frames: list[FrameResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CorrectionIn(BaseModel):
    frame_id: str
    landmarks: LandmarksModel


class CorrectionOut(BaseModel):
    job_id: str
    frame: FrameResult
    aggregate: AggregateResult


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    model_version: Optional[str] = None


class ModelConfig(BaseModel):
    """Tunable pipeline parameters (all optional; sensible defaults)."""

    model_config = ConfigDict(extra="forbid")

    sample_fps: Optional[float] = Field(None, gt=0, le=30)
    max_frames: Optional[int] = Field(None, gt=0, le=1000)
    min_laplacian_var: Optional[float] = Field(None, ge=0)
    max_yaw_deg: Optional[float] = Field(None, ge=0)
    max_pitch_deg: Optional[float] = Field(None, ge=0)
    max_roll_deg: Optional[float] = Field(None, ge=0)
    median_window: Optional[int] = Field(None, ge=1, le=31)

"""End-to-end analysis pipeline.

Implements ARCHITECTURE.md section 6 (runtime data flow) for the MVP
baseline: decode -> sample -> quality gate -> MediaPipe landmarks ->
validation -> temporal filter -> measurement -> aggregation -> report.

Custom PyTorch heatmap inference (landmark_model.py) is wired in when a
checkpoint is present; otherwise the MediaPipe baseline candidates are
used and every result carries explicit warnings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_PITCH_DEG,
    DEFAULT_MAX_ROLL_DEG,
    DEFAULT_MAX_YAW_DEG,
    DEFAULT_MEDIAN_WINDOW,
    DEFAULT_MIN_LAPLACIAN_VAR,
    DEFAULT_SAMPLE_FPS,
    MODEL_VERSION,
)
from ..schemas import (
    AnalysisStatus,
    FrameQuality,
    FrameRejection,
    FrameResult,
    LandmarkSource,
    LandmarksModel,
)
from . import face_detector
from .landmark_validation import validate_frame
from .measurement import Landmarks, Point
from .quality_gate import QualityConfig, evaluate_frame
from .report_builder import (
    build_aggregate,
    build_quality_summary,
    default_warnings,
    landmarks_to_schema,
    measurements_from_landmarks,
    model_info,
    representative_frame_id,
    schema_to_landmarks,
)
from .temporal_filter import filter_landmark_series
from .video_reader import sample_frames

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    sample_fps: float = DEFAULT_SAMPLE_FPS
    max_frames: int = DEFAULT_MAX_FRAMES
    min_laplacian_var: float = DEFAULT_MIN_LAPLACIAN_VAR
    max_yaw_deg: float = DEFAULT_MAX_YAW_DEG
    max_pitch_deg: float = DEFAULT_MAX_PITCH_DEG
    max_roll_deg: float = DEFAULT_MAX_ROLL_DEG
    median_window: int = DEFAULT_MEDIAN_WINDOW
    min_frame_side: int = 240

    def quality_config(self) -> QualityConfig:
        return QualityConfig(
            min_laplacian_var=self.min_laplacian_var,
            min_frame_side=self.min_frame_side,
            max_yaw_deg=self.max_yaw_deg,
            max_pitch_deg=self.max_pitch_deg,
            max_roll_deg=self.max_roll_deg,
        )


@dataclass
class _Candidate:
    frame_index: int
    timestamp_ms: int
    image_shape: tuple[int, int]  # h, w
    landmarks: Optional[Landmarks]
    face_box: Optional[tuple[float, float, float, float]]
    blur_variance: float
    yaw_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    quality: FrameQuality
    rejection: Optional[FrameRejection]
    warnings: list[str] = field(default_factory=list)


def run_pipeline(video_path: str, cfg: Optional[PipelineConfig] = None) -> dict:
    """Analyse one video and return an AnalysisResult-shaped dict."""
    cfg = cfg or PipelineConfig()
    qcfg = cfg.quality_config()
    warnings = default_warnings()

    candidates: list[_Candidate] = []
    total_sampled = 0

    for frame in sample_frames(video_path, cfg.sample_fps, cfg.max_frames):
        total_sampled += 1
        verdict = evaluate_frame(frame.image, None, qcfg, pose_available=False)
        # Re-evaluate with landmarks if face is present.
        lm_raw = face_detector.detect(frame.image, frame.timestamp_ms)
        if lm_raw is not None:
            pixel_xy = lm_raw.all_landmarks_xy
            verdict = evaluate_frame(frame.image, pixel_xy, qcfg, pose_available=True)
        else:
            verdict = evaluate_frame(frame.image, None, qcfg, pose_available=False)

        if not verdict.passed:
            primary = verdict.reasons[0] if verdict.reasons else "unknown"
            candidates.append(
                _Candidate(
                    frame_index=frame.index,
                    timestamp_ms=frame.timestamp_ms,
                    image_shape=frame.image.shape[:2],
                    landmarks=None,
                    face_box=None,
                    blur_variance=verdict.blur_variance,
                    yaw_deg=verdict.pose.yaw_deg if verdict.pose else None,
                    pitch_deg=verdict.pose.pitch_deg if verdict.pose else None,
                    roll_deg=verdict.pose.roll_deg if verdict.pose else None,
                    quality=FrameQuality.rejected,
                    rejection=FrameRejection(reason=primary, detail=";".join(verdict.reasons)),
                )
            )
            continue

        assert lm_raw is not None
        lm = Landmarks(
            nasion=Point(*lm_raw.nasion),
            menton=Point(*lm_raw.menton),
            zygion_right=Point(*lm_raw.zygion_right),
            zygion_left=Point(*lm_raw.zygion_left),
        )
        v = validate_frame(lm, face_box=lm_raw.face_box)
        if not v.ok:
            candidates.append(
                _Candidate(
                    frame_index=frame.index,
                    timestamp_ms=frame.timestamp_ms,
                    image_shape=frame.image.shape[:2],
                    landmarks=lm,
                    face_box=lm_raw.face_box,
                    blur_variance=verdict.blur_variance,
                    yaw_deg=verdict.pose.yaw_deg if verdict.pose else None,
                    pitch_deg=verdict.pose.pitch_deg if verdict.pose else None,
                    roll_deg=verdict.pose.roll_deg if verdict.pose else None,
                    quality=FrameQuality.rejected,
                    rejection=FrameRejection(
                        reason="geometry_invalid", detail=";".join(v.errors)
                    ),
                    warnings=list(v.warnings),
                )
            )
            continue

        # Accepted (or flagged if face confidence is low).
        quality = FrameQuality.accepted
        rej = None
        if lm_raw.face_confidence < 0.7:
            quality = FrameQuality.flagged
            rej = FrameRejection(reason="low_face_confidence",
                                 detail=f"{lm_raw.face_confidence:.2f}")

        candidates.append(
            _Candidate(
                frame_index=frame.index,
                timestamp_ms=frame.timestamp_ms,
                image_shape=frame.image.shape[:2],
                landmarks=lm,
                face_box=lm_raw.face_box,
                blur_variance=verdict.blur_variance,
                yaw_deg=verdict.pose.yaw_deg if verdict.pose else None,
                pitch_deg=verdict.pose.pitch_deg if verdict.pose else None,
                roll_deg=verdict.pose.roll_deg if verdict.pose else None,
                quality=quality,
                rejection=rej,
                warnings=list(v.warnings),
            )
        )

    # --- Temporal filter over accepted+flagged landmarks -----------------
    usable = [c for c in candidates if c.landmarks is not None]
    if len(usable) >= 3 and cfg.median_window >= 3:
        series = filter_landmark_series(
            [c.landmarks for c in usable], cfg.median_window
        )
        for c, lm in zip(usable, series):
            c.landmarks = lm  # type: ignore[assignment]

    # --- Measurement per frame -------------------------------------------
    frame_results: list[FrameResult] = []
    for c in candidates:
        measurements = None
        if c.landmarks is not None and c.quality != FrameQuality.rejected:
            try:
                measurements = measurements_from_landmarks(c.landmarks)
            except Exception:
                measurements = None
        lm_schema = (
            landmarks_to_schema(c.landmarks, confidence=1.0, source=LandmarkSource.model.value)
            if c.landmarks is not None
            else LandmarksModel()
        )
        frame_results.append(
            FrameResult(
                frame_id=f"f_{c.frame_index:06d}",
                frame_index=c.frame_index,
                timestamp_ms=c.timestamp_ms,
                status=c.quality,
                landmarks=lm_schema,
                measurements=measurements,
                rejection=c.rejection,
                blur_variance=c.blur_variance,
                yaw_deg=c.yaw_deg,
                pitch_deg=c.pitch_deg,
                roll_deg=c.roll_deg,
                face_box=c.face_box,
            )
        )

    quality_summary = build_quality_summary(frame_results)
    aggregate = build_aggregate(frame_results)
    rep_id = representative_frame_id(frame_results)

    if quality_summary.accepted == 0:
        warnings.append(
            "No frames passed the quality gate; the report is indicative only."
        )
    if aggregate.facial_index is not None and aggregate.accepted_frame_count < 5:
        warnings.append(
            f"Only {aggregate.accepted_frame_count} accepted frame(s); "
            "aggregate may be unstable."
        )

    return {
        "status": AnalysisStatus.completed.value,
        "model": model_info().model_dump(),
        "quality": quality_summary.model_dump(),
        "aggregate": aggregate.model_dump(),
        "representative_frame_id": rep_id,
        "frames": [f.model_dump() for f in frame_results],
        "warnings": warnings,
        "config": {
            "sample_fps": cfg.sample_fps,
            "max_frames": cfg.max_frames,
            "min_laplacian_var": cfg.min_laplacian_var,
            "max_yaw_deg": cfg.max_yaw_deg,
            "max_pitch_deg": cfg.max_pitch_deg,
            "max_roll_deg": cfg.max_roll_deg,
            "median_window": cfg.median_window,
        },
        "total_sampled": total_sampled,
        "model_version": MODEL_VERSION,
    }

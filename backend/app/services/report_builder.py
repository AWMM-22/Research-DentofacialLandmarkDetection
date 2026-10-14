"""Report assembly: per-frame results -> aggregate summary.

Implements ARCHITECTURE.md section 6E: calculate H/W/FI per accepted frame,
aggregate with the median FI, assign category from the *unrounded* aggregate,
and retain per-frame measurements for audit.
"""

from __future__ import annotations

from typing import Optional

from ..config import MODEL_VERSION
from ..schemas import (
    AggregateResult,
    FrameMeasurements,
    FrameQuality,
    FrameResult,
    LandmarksModel,
    LandmarkPoint,
    ModelInfo,
    QualitySummary,
)
from .measurement import (
    Landmarks,
    Point,
    aggregate_fi,
    classify,
    measure,
)


def landmarks_to_schema(lm: Landmarks, confidence: float, source: str) -> LandmarksModel:
    return LandmarksModel(
        nasion=LandmarkPoint(x=lm.nasion.x, y=lm.nasion.y, confidence=confidence, source=source),
        menton=LandmarkPoint(x=lm.menton.x, y=lm.menton.y, confidence=confidence, source=source),
        zygion_right=LandmarkPoint(x=lm.zygion_right.x, y=lm.zygion_right.y, confidence=confidence, source=source),
        zygion_left=LandmarkPoint(x=lm.zygion_left.x, y=lm.zygion_left.y, confidence=confidence, source=source),
    )


def schema_to_landmarks(lm: LandmarksModel) -> Optional[Landmarks]:
    if lm.nasion is None or lm.menton is None or lm.zygion_right is None or lm.zygion_left is None:
        return None
    return Landmarks(
        nasion=Point(lm.nasion.x, lm.nasion.y),
        menton=Point(lm.menton.x, lm.menton.y),
        zygion_right=Point(lm.zygion_right.x, lm.zygion_right.y),
        zygion_left=Point(lm.zygion_left.x, lm.zygion_left.y),
    )


def build_quality_summary(frames: list[FrameResult]) -> QualitySummary:
    summary = QualitySummary(sampled=len(frames))
    for f in frames:
        if f.status == FrameQuality.accepted:
            summary.accepted += 1
        elif f.status == FrameQuality.rejected:
            summary.rejected += 1
            if f.rejection is not None:
                key = f.rejection.reason
                summary.rejection_reasons[key] = summary.rejection_reasons.get(key, 0) + 1
        elif f.status == FrameQuality.flagged:
            summary.flagged += 1
    return summary


def build_aggregate(frames: list[FrameResult]) -> AggregateResult:
    """Median FI over *accepted* frames only (AGENT_PLAN.md section 7)."""
    accepted = [f for f in frames if f.status == FrameQuality.accepted and f.measurements is not None]
    fis = [f.measurements.facial_index for f in accepted if f.measurements is not None]
    heights = [f.measurements.height for f in accepted if f.measurements is not None]
    widths = [f.measurements.width for f in accepted if f.measurements is not None]

    fi = aggregate_fi(fis)
    category = classify(fi) if fi is not None else "Outside the specified reference ranges"
    median_h = aggregate_fi(heights)  # median helper works for any list
    median_w = aggregate_fi(widths)

    return AggregateResult(
        facial_index=fi,
        category=category,
        median_height=median_h,
        median_width=median_w,
        accepted_frame_count=len(accepted),
    )


def representative_frame_id(frames: list[FrameResult]) -> Optional[str]:
    """Pick the accepted frame whose FI is closest to the aggregate median."""
    accepted = [f for f in frames if f.status == FrameQuality.accepted and f.measurements is not None]
    if not accepted:
        # Fall back to first flagged frame with landmarks for display.
        for f in frames:
            if f.status == FrameQuality.flagged and f.measurements is not None:
                return f.frame_id
        return None
    agg = build_aggregate(frames)
    if agg.facial_index is None:
        return accepted[0].frame_id
    best = min(accepted, key=lambda f: abs(f.measurements.facial_index - agg.facial_index))
    return best.frame_id


def default_warnings() -> list[str]:
    return [
        "Baseline MediaPipe landmarks are candidates for visualisation only; "
        "they are NOT validated anthropometric labels (see ARCHITECTURE.md section 4).",
        "Menton and zygion approximations must be manually confirmed or produced "
        "by the custom heatmap model before any measurement is trusted.",
        "This system is a measurement/classification prototype, not a medical "
        "diagnostic device. Results require expert validation.",
    ]


def model_info() -> ModelInfo:
    return ModelInfo(
        name="MediaPipe Face Landmarker",
        version=MODEL_VERSION,
        note="Baseline only; generic mesh, not anthropometric ground truth.",
    )


def measurements_from_landmarks(lm: Landmarks) -> FrameMeasurements:
    m = measure(lm)
    return FrameMeasurements(height=m.height, width=m.width, facial_index=m.facial_index)

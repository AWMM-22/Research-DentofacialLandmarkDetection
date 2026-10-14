"""MediaPipe Face Landmarker baseline service.

Implements ARCHITECTURE.md section 3 "Baseline — MediaPipe Face Landmarker":
face detection/tracking and generic mesh landmarks for initialisation and
visualisation only.

CRITICAL (AGENT_PLAN.md section 5, ARCHITECTURE.md section 4): the four
anthropometric target points are *approximated* from mesh indices listed in
config.MEDIAPIPE_CANDIDATE_INDICES. These candidates are NOT validated
anthropometric labels. Zygion especially is only an outer face-contour
proxy. Every result returned by this module is marked as model-sourced and
must be confirmed manually or replaced by the custom heatmap model before
being treated as ground truth.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import (
    FACE_LANDMARKER_MODEL,
    MEDIAPIPE_CANDIDATE_INDICES,
    MODEL_VERSION,
)


class FaceDetectorError(RuntimeError):
    pass


@dataclass(frozen=True)
class BaselineLandmarks:
    """Four candidate points in original-frame pixel coordinates."""

    nasion: tuple[float, float]
    menton: tuple[float, float]
    zygion_right: tuple[float, float]
    zygion_left: tuple[float, float]
    face_box: tuple[float, float, float, float]  # x, y, w, h in pixels
    all_landmarks_xy: np.ndarray                 # (N, 2) pixels
    face_confidence: float                       # detector score if available


# Lazy, thread-safe singleton so FastAPI workers share one model instance.
_lock = threading.Lock()
_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    with _lock:
        if _landmarker is not None:
            return _landmarker
        if not FACE_LANDMARKER_MODEL.exists():
            raise FaceDetectorError(
                f"Face Landmarker model not found: {FACE_LANDMARKER_MODEL}. "
                "Download face_landmarker.task into backend/app/models/."
            )
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
            )
        except ImportError as exc:  # pragma: no cover
            raise FaceDetectorError(
                "mediapipe is not installed; install backend/requirements.txt"
            ) from exc

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarker = FaceLandmarker.create_from_options(options)
        return _landmarker


def model_version() -> str:
    return MODEL_VERSION


def model_loaded() -> bool:
    return _landmarker is not None


def _to_mp_image(bgr: np.ndarray):
    import mediapipe as mp

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def detect(
    bgr: np.ndarray,
    timestamp_ms: int,
) -> Optional[BaselineLandmarks]:
    """Run MediaPipe on one BGR frame (VIDEO mode, monotonically increasing
    timestamps required). Returns None when no face is found.
    """
    landmarker = _get_landmarker()
    h, w = bgr.shape[:2]
    mp_image = _to_mp_image(bgr)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]
    xs = np.array([lm.x for lm in face], dtype=np.float64) * w
    ys = np.array([lm.y for lm in face], dtype=np.float64) * h
    coords = np.stack([xs, ys], axis=1)

    idx = MEDIAPIPE_CANDIDATE_INDICES

    def pt(name: str) -> tuple[float, float]:
        i = idx[name]
        return (float(coords[i, 0]), float(coords[i, 1]))

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    face_box = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))

    # Detector confidence: tasks API exposes face_presence_scores per face.
    conf = 1.0
    if getattr(result, "face_presence_scores", None):
        try:
            conf = float(result.face_presence_scores[0])
        except (IndexError, TypeError, ValueError):
            conf = 1.0

    return BaselineLandmarks(
        nasion=pt("nasion"),
        menton=pt("menton"),
        zygion_right=pt("zygion_right"),
        zygion_left=pt("zygion_left"),
        face_box=face_box,
        all_landmarks_xy=coords,
        face_confidence=conf,
    )


def reset() -> None:
    """Release the singleton (used by tests)."""
    global _landmarker
    with _lock:
        if _landmarker is not None:
            try:
                _landmarker.close()
            except Exception:
                pass
        _landmarker = None

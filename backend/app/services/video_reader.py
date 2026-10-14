"""OpenCV video decoding and timestamp-preserving frame sampling.

Implements ARCHITECTURE.md section 6B: decode frames, preserve timestamps,
sample at a configurable rate to limit compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReadError(RuntimeError):
    """Raised when a video cannot be opened or decoded."""


@dataclass(frozen=True)
class VideoMeta:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_ms: int


@dataclass(frozen=True)
class Frame:
    """One decoded frame with its original timestamp."""

    index: int          # 0-based source frame index
    timestamp_ms: int
    image: np.ndarray   # BGR, HxWx3


def read_meta(path: str | Path) -> VideoMeta:
    """Open the video and return basic metadata without decoding frames."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError(f"Cannot open video: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0:
            raise VideoReadError(
                f"Invalid video metadata (fps={fps}, frames={frame_count}): {path}"
            )
        duration_ms = int(round(frame_count / fps * 1000))
        return VideoMeta(
            path=str(path),
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration_ms=duration_ms,
        )
    finally:
        cap.release()


def sample_frames(
    path: str | Path,
    sample_fps: float,
    max_frames: int,
) -> Iterator[Frame]:
    """Yield frames at approximately `sample_fps`, capped at `max_frames`.

    Timestamps are derived from the source frame index and source fps so the
    original temporal position is preserved. Decoding stops once enough
    frames have been collected.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError(f"Cannot open video: {path}")
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if src_fps <= 0:
            raise VideoReadError(f"Invalid source fps: {src_fps}")

        step = max(1, int(round(src_fps / sample_fps)))
        yielded = 0
        index = 0
        while yielded < max_frames:
            ok, image = cap.read()
            if not ok:
                break
            if index % step == 0:
                timestamp_ms = int(round(index / src_fps * 1000))
                yield Frame(index=index, timestamp_ms=timestamp_ms, image=image)
                yielded += 1
            index += 1
    finally:
        cap.release()

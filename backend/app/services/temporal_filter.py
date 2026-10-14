"""Temporal association and jitter filtering.

Implements ARCHITECTURE.md section 6D: associate the same face across
frames and apply a short-window median filter to reduce random jitter.

NOTE (AGENT_PLAN.md section 7): filtering cannot correct a systematically
wrong landmark. The filter reduces *random* frame-to-frame noise only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .measurement import Landmarks, Point


def median_filter_series(
    coords: np.ndarray,
    window: int,
) -> np.ndarray:
    """Centre-aligned moving median along axis 0.

    `coords` is (T, 2). Edge frames use a truncated window (partial support)
    rather than padding with edge values, so boundaries are not artificially
    frozen. `window` must be a positive odd integer.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    t = coords.shape[0]
    if t == 0 or window == 1:
        return coords.copy()

    half = window // 2
    out = np.empty_like(coords)
    for i in range(t):
        lo = max(0, i - half)
        hi = min(t, i + half + 1)
        out[i] = np.median(coords[lo:hi], axis=0)
    return out


def filter_landmark_series(
    series: Sequence[Landmarks],
    window: int,
) -> list[Landmarks]:
    """Apply the median filter to a sequence of Landmarks."""
    if not series:
        return []
    arr = np.array(
        [
            [
                [lm.nasion.x, lm.nasion.y],
                [lm.menton.x, lm.menton.y],
                [lm.zygion_right.x, lm.zygion_right.y],
                [lm.zygion_left.x, lm.zygion_left.y],
            ]
            for lm in series
        ],
        dtype=np.float64,
    )  # (T, 4, 2)
    t = arr.shape[0]
    filtered = np.empty_like(arr)
    for k in range(4):
        filtered[:, k, :] = median_filter_series(arr[:, k, :], window)

    out: list[Landmarks] = []
    for i in range(t):
        out.append(
            Landmarks(
                nasion=Point(float(filtered[i, 0, 0]), float(filtered[i, 0, 1])),
                menton=Point(float(filtered[i, 1, 0]), float(filtered[i, 1, 1])),
                zygion_right=Point(float(filtered[i, 2, 0]), float(filtered[i, 2, 1])),
                zygion_left=Point(float(filtered[i, 3, 0]), float(filtered[i, 3, 1])),
            )
        )
    return out

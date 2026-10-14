"""Measurement engine: distances, facial height/width, FI, and category rules.

Pure functions with no I/O. Implements AGENT_PLAN.md section 4 and
ARCHITECTURE.md section 5 exactly:

    d(A, B)   = sqrt((x_A - x_B)^2 + (y_A - y_B)^2)
    H         = d(N', Me')
    W         = d(Zy_r, Zy_l)
    FI        = (H / W) * 100

Classification uses half-open intervals on the *unrounded* FI. Values
outside all configured bands are never forced into a category.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..config import FI_OUTSIDE_LABEL, FI_RANGES


@dataclass(frozen=True)
class Point:
    """2D image coordinate in pixels."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError("Point coordinates must be finite numbers")


@dataclass(frozen=True)
class Landmarks:
    """The four anthropometric targets required for the Facial Index."""

    nasion: Point
    menton: Point
    zygion_right: Point
    zygion_left: Point


@dataclass(frozen=True)
class Measurement:
    """Result of measuring one set of landmarks."""

    height: float          # H, pixels
    width: float           # W, pixels
    facial_index: float    # FI, unitless * 100


class MeasurementError(ValueError):
    """Raised when landmarks cannot produce a valid measurement."""


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points, in the input coordinate units."""
    return math.hypot(a.x - b.x, a.y - b.y)


def facial_height(lm: Landmarks) -> float:
    """H = d(N', Me'). Pixels."""
    return distance(lm.nasion, lm.menton)


def bizygomatic_width(lm: Landmarks) -> float:
    """W = d(Zy_r, Zy_l). Pixels."""
    return distance(lm.zygion_right, lm.zygion_left)


def facial_index(height: float, width: float) -> float:
    """FI = (H / W) * 100.

    Raises MeasurementError if width is not strictly positive.
    """
    if not math.isfinite(width) or width <= 0:
        raise MeasurementError(
            "Bizygomatic width must be a positive finite number; "
            f"got {width!r}"
        )
    if not math.isfinite(height) or height < 0:
        raise MeasurementError(
            "Facial height must be a non-negative finite number; "
            f"got {height!r}"
        )
    return (height / width) * 100.0


def measure(lm: Landmarks) -> Measurement:
    """Compute H, W, and FI from a full landmark set."""
    h = facial_height(lm)
    w = bizygomatic_width(lm)
    fi = facial_index(h, w)
    return Measurement(height=h, width=w, facial_index=fi)


def classify(fi: float) -> str:
    """Classify an *unrounded* FI using half-open [lower, upper) intervals.

    Values outside every configured band return the outside label and are
    never forced into a category (AGENT_PLAN.md section 4).
    """
    if not math.isfinite(fi):
        raise MeasurementError(f"FI must be finite to classify; got {fi!r}")
    for lower, upper, label in FI_RANGES:
        if lower <= fi < upper:
            return label
    return FI_OUTSIDE_LABEL


def aggregate_fi(fis: list[float]) -> Optional[float]:
    """Robust aggregate of per-frame FI values: the median.

    Returns None when the list is empty. Full precision is preserved;
    callers round only for display.
    """
    if not fis:
        return None
    ordered = sorted(fis)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0

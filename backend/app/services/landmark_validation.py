"""Landmark and geometry validation.

Implements ARCHITECTURE.md section 6D: check that all four points exist,
bizygomatic width is positive, and geometry is plausible relative to the
face bounding box. Frames that fail validation are flagged or rejected —
they are never silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import (
    MAX_HEIGHT_TO_FACE_RATIO,
    MAX_H_TO_W_RATIO,
    MIN_H_TO_W_RATIO,
    MIN_WIDTH_TO_FACE_RATIO,
)
from .measurement import Landmarks, Point, distance


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_completeness(lm: Landmarks) -> list[str]:
    """All four points must exist; Point construction already rejects NaN."""
    # Landmarks is a frozen dataclass of Points; constructing it guarantees
    # presence. Kept as an explicit step for API clarity / future optional
    # fields (e.g. confidence=None means "unobservable").
    return []


def validate_width(lm: Landmarks) -> list[str]:
    w = distance(lm.zygion_right, lm.zygion_left)
    if w <= 0:
        return ["width_not_positive"]
    return []


def validate_against_face_box(
    lm: Landmarks,
    face_box: tuple[float, float, float, float],
) -> tuple[list[str], list[str]]:
    """Plausibility checks relative to the detected face bounding box.

    Returns (errors, warnings). Fails closed on clearly impossible
    geometry; warns on values that are unusual but not impossible.
    """
    errors: list[str] = []
    warnings: list[str] = []
    bx, by, bw, bh = face_box
    if bw <= 0 or bh <= 0:
        return (["invalid_face_box"], warnings)

    h = distance(lm.nasion, lm.menton)
    w = distance(lm.zygion_right, lm.zygion_left)

    # Width must span a reasonable fraction of the face box width.
    if w < MIN_WIDTH_TO_FACE_RATIO * bw:
        errors.append(
            f"width_too_small_vs_face:{w / bw:.2f}<{MIN_WIDTH_TO_FACE_RATIO}"
        )

    # Height must not exceed the face box by a large margin.
    if h > MAX_HEIGHT_TO_FACE_RATIO * bh:
        errors.append(
            f"height_too_large_vs_face:{h / bh:.2f}>{MAX_HEIGHT_TO_FACE_RATIO}"
        )

    # H/W ratio sanity band (catches swapped points, wild outliers).
    if w > 0:
        ratio = h / w
        if ratio < MIN_H_TO_W_RATIO:
            errors.append(f"h_over_w_too_small:{ratio:.2f}<{MIN_H_TO_W_RATIO}")
        elif ratio > MAX_H_TO_W_RATIO:
            errors.append(f"h_over_w_too_large:{ratio:.2f}>{MAX_H_TO_W_RATIO}")

    # Vertical ordering: menton should be below nasion in image coords.
    if lm.menton.y <= lm.nasion.y:
        errors.append("menton_above_nasion")

    # Left/right ordering (subject's left is typically screen-right in a
    # non-mirrored selfie, but we only warn on apparent reversal because
    # camera mirroring varies).
    if lm.zygion_left.x < lm.zygion_right.x:
        warnings.append("zygion_left_screen_left")

    return errors, warnings


def validate_frame(
    lm: Optional[Landmarks],
    face_box: Optional[tuple[float, float, float, float]] = None,
) -> ValidationResult:
    """Full validation for one frame's landmark set."""
    if lm is None:
        return ValidationResult(ok=False, errors=["missing_landmarks"])

    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(validate_completeness(lm))
    errors.extend(validate_width(lm))

    if face_box is not None:
        e2, w2 = validate_against_face_box(lm, face_box)
        errors.extend(e2)
        warnings.extend(w2)

    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)

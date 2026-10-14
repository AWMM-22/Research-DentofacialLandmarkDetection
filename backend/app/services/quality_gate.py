"""Frame quality gate: blur, resolution, and approximate head pose.

Implements ARCHITECTURE.md section 6B. Frames that fail any gate are
rejected with an explicit reason so the pipeline never fabricates a
measurement from unsuitable input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    min_laplacian_var: float = 40.0
    min_frame_side: int = 240
    max_yaw_deg: float = 18.0
    max_pitch_deg: float = 18.0
    max_roll_deg: float = 15.0


@dataclass(frozen=True)
class PoseEstimate:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass(frozen=True)
class QualityVerdict:
    passed: bool
    blur_variance: float
    pose: Optional[PoseEstimate]
    reasons: list[str]


def blur_variance(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian. Low values indicate blur."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_resolution(image_bgr: np.ndarray, min_side: int) -> list[str]:
    h, w = image_bgr.shape[:2]
    if min(h, w) < min_side:
        return [f"resolution_too_small:{w}x{h}"]
    return []


# MediaPipe 478-point indices used for coarse head-pose estimation.
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_LEFT_EYE_INNER = 133
_RIGHT_EYE_INNER = 362
_FOREHEAD = 10          # approximate glabella / upper forehead
_CHEEK_RIGHT = 234      # outer face contour (subject's right in canonical image)
_CHEEK_LEFT = 454


def estimate_pose(
    landmarks_xy: np.ndarray,
    image_width: int,
    image_height: int,
) -> Optional[PoseEstimate]:
    """Estimate coarse yaw / pitch / roll from 2D landmarks.

    Geometric method (more robust than solvePnP with a generic 3D model
    for near-frontal gating):

      roll  — angle of the eye-corner line from horizontal
      yaw   — horizontal offset of the nose tip from the face midline,
              normalised by half face width
      pitch — vertical position of the nose tip on the eye→chin axis,
              mapped to a rough degree scale

    Returns None when required indices are missing.
    """
    if landmarks_xy.shape[0] <= max(_CHEEK_LEFT, _CHEEK_RIGHT, _FOREHEAD, _CHIN):
        return None

    le = landmarks_xy[_LEFT_EYE_OUTER]
    re = landmarks_xy[_RIGHT_EYE_OUTER]
    nose = landmarks_xy[_NOSE_TIP]
    chin = landmarks_xy[_CHIN]
    cr = landmarks_xy[_CHEEK_RIGHT]
    cl = landmarks_xy[_CHEEK_LEFT]

    # ---- roll: eye-line angle from horizontal (image coords, y down) ----
    dx = float(re[0] - le[0])
    dy = float(re[1] - le[1])
    roll = float(np.degrees(np.arctan2(dy, dx)))
    # Fold into [-90, 90]; a near-upright face gives ~0 regardless of
    # which detected eye is "left" in image space.
    if roll > 90.0:
        roll -= 180.0
    elif roll < -90.0:
        roll += 180.0

    # ---- yaw: nose offset from cheek midline, as fraction of half-width
    face_mid_x = float((cr[0] + cl[0]) / 2.0)
    half_w = float(abs(cl[0] - cr[0]) / 2.0)
    if half_w < 1.0:
        yaw = 0.0
    else:
        # Map lateral offset to degrees: atan of offset / half-width.
        yaw = float(np.degrees(np.arctan2(float(nose[0]) - face_mid_x, half_w)))

    # ---- pitch: nose vertical position between eye line and chin ----
    eye_mid_y = float((le[1] + re[1]) / 2.0)
    face_h = float(chin[1] - eye_mid_y)
    if face_h < 1.0:
        pitch = 0.0
    else:
        # ratio ~0.35–0.55 for a typical upright frontal face.
        ratio = float(nose[1] - eye_mid_y) / face_h
        pitch = (ratio - 0.45) * 160.0  # rough degree mapping

    return PoseEstimate(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)


def check_pose(
    pose: Optional[PoseEstimate],
    cfg: QualityConfig,
) -> list[str]:
    if pose is None:
        return ["pose_unavailable"]
    reasons: list[str] = []
    if abs(pose.yaw_deg) > cfg.max_yaw_deg:
        reasons.append(f"yaw_out_of_range:{pose.yaw_deg:.1f}")
    if abs(pose.pitch_deg) > cfg.max_pitch_deg:
        reasons.append(f"pitch_out_of_range:{pose.pitch_deg:.1f}")
    if abs(pose.roll_deg) > cfg.max_roll_deg:
        reasons.append(f"roll_out_of_range:{pose.roll_deg:.1f}")
    return reasons


def evaluate_frame(
    image_bgr: np.ndarray,
    landmarks_xy: Optional[np.ndarray],
    cfg: QualityConfig,
    pose_available: bool = True,
) -> QualityVerdict:
    """Run all quality checks for one frame.

    `landmarks_xy` is the full (N, 2) MediaPipe landmark array in pixel
    coordinates, or None when no face was detected.
    """
    reasons: list[str] = []
    reasons.extend(check_resolution(image_bgr, cfg.min_frame_side))

    bvar = blur_variance(image_bgr)
    if bvar < cfg.min_laplacian_var:
        reasons.append(f"blur:{bvar:.1f}")

    pose: Optional[PoseEstimate] = None
    if landmarks_xy is None:
        reasons.append("no_face")
    elif pose_available:
        h, w = image_bgr.shape[:2]
        # MediaPipe normalises to [0,1]; convert to pixels for solvePnP.
        pixel_xy = landmarks_xy.copy()
        if pixel_xy.max() <= 1.5:
            pixel_xy[:, 0] *= w
            pixel_xy[:, 1] *= h
        pose = estimate_pose(pixel_xy, w, h)
        reasons.extend(check_pose(pose, cfg))

    return QualityVerdict(
        passed=len(reasons) == 0,
        blur_variance=bvar,
        pose=pose,
        reasons=reasons,
    )

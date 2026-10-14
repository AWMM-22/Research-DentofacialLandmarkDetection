"""Central configuration for the Facial Index analysis pipeline.

All thresholds are prototype defaults and must be calibrated with expert
input before any real-world claim of accuracy (AGENT_PLAN.md section 3,
"Initial acceptance targets").
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BACKEND_DIR / "app"
MODELS_DIR = APP_DIR / "models"
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
DATA_DIR = BACKEND_DIR.parent / "data"
WORKSPACE_DIR = DATA_DIR / "workspace"

# ---------------------------------------------------------------------------
# FI classification (AGENT_PLAN.md section 4 / ARCHITECTURE.md section 5)
# Half-open intervals: [lower, upper). Classification uses the *unrounded*
# FI; rounding is display-only.
# ---------------------------------------------------------------------------
FI_RANGES: list[tuple[float, float, str]] = [
    (80.0, 85.0, "Euryprosopic"),
    (85.0, 90.0, "Mesoprosopic"),
    (90.0, 95.0, "Leptoprosopic"),
]
FI_OUTSIDE_LABEL = "Outside the specified reference ranges"

# ---------------------------------------------------------------------------
# Video / frame sampling
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_FPS = 2.0          # frames per second to analyse
DEFAULT_MAX_FRAMES = 90           # hard cap on analysed frames
DEFAULT_MIN_FRAME_SIDE = 240      # reject frames smaller than this (px)

# ---------------------------------------------------------------------------
# Quality gate (prototype defaults; calibrate with expert)
# ---------------------------------------------------------------------------
DEFAULT_MIN_LAPLACIAN_VAR = 40.0  # blur gate: variance of Laplacian
DEFAULT_MAX_YAW_DEG = 18.0        # near-frontal gate
DEFAULT_MAX_PITCH_DEG = 18.0
DEFAULT_MAX_ROLL_DEG = 15.0

# ---------------------------------------------------------------------------
# Landmark validation (geometry plausibility)
# ---------------------------------------------------------------------------
# Minimum bizygomatic width as a fraction of the face bounding-box width.
# A predicted width far below this is geometrically implausible.
MIN_WIDTH_TO_FACE_RATIO = 0.35
# Maximum vertical extent (nasion -> menton) relative to face-box height.
MAX_HEIGHT_TO_FACE_RATIO = 1.10
# Minimum ratio of facial height to bizygomatic width (sanity floor).
MIN_H_TO_W_RATIO = 0.50
# Maximum ratio (sanity ceiling).
MAX_H_TO_W_RATIO = 1.60

# ---------------------------------------------------------------------------
# Temporal filter
# ---------------------------------------------------------------------------
DEFAULT_MEDIAN_WINDOW = 5         # odd window length (frames each side + centre)

# ---------------------------------------------------------------------------
# MediaPipe baseline landmark indices (478-point topology, tasks API)
#
# IMPORTANT (ARCHITECTURE.md section 4, AGENT_PLAN.md section 5): these are
# *candidates* for visualisation / initialisation only. They are NOT
# validated anthropometric labels. Zygion especially is only approximated
# by the outer face-contour points below. Manual confirmation or the custom
# heatmap model is required before trusting measurements.
# ---------------------------------------------------------------------------
MEDIAPIPE_CANDIDATE_INDICES: dict[str, int] = {
    "nasion": 168,       # root of nose, between the eyes
    "menton": 152,       # lowest point of chin contour
    "zygion_right": 234,  # right outer face contour (subject's right)
    "zygion_left": 454,   # left outer face contour (subject's left)
}

MODEL_VERSION = "mediapipe-face-landmarker-float16-1"

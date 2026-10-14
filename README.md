# Facial Index Video Classification

Local-first prototype that estimates the **Facial Index (FI)** from a
webcam stream or a prerecorded video, then classifies the face into a
proportional category (Euryprosopic / Mesoprosopic / Leptoprosopic).

> **This is a measurement and classification prototype, not a medical
> diagnostic device.** Outputs depend on camera perspective, pose, image
> quality, the chosen anthropometric definitions, and the accuracy of the
> landmark annotations. Validate against an expert reference before any
> clinical or forensic use.

Design documents: [`Research/AGENT_PLAN.md`](Research/AGENT_PLAN.md) ·
[`Research/ARCHITECTURE.md`](Research/ARCHITECTURE.md)

---

## Table of contents

1. [What is the Facial Index?](#1-what-is-the-facial-index)
2. [The four landmarks](#2-the-four-landmarks)
3. [How the pipeline works](#3-how-the-pipeline-works)
4. [Quick start](#4-quick-start)
5. [Project structure](#5-project-structure)
6. [API reference](#6-api-reference)
7. [Webcam demo](#7-webcam-demo)
8. [Web UI](#8-web-ui)
9. [Models](#9-models)
10. [Configuration](#10-configuration)
11. [Testing](#11-testing)
12. [Important limitations](#12-important-limitations)
13. [Privacy](#13-privacy)
14. [Roadmap](#14-roadmap)
15. [Licences](#15-licences)

---

## 1. What is the Facial Index?

The Facial Index is the ratio of facial height to bizygomatic (cheek)
width, expressed as a percentage:

```
d(A, B)  = sqrt((x_A − x_B)² + (y_A − y_B)²)

H = d(N′, Me′)          facial height        (nasion → menton)
W = d(Zyᵣ, Zyₗ)         bizygomatic width    (right zygion → left zygion)

FI = (H / W) × 100
```

Because both H and W are measured in the same image units (pixels), the
units cancel in the ratio. This does **not** remove perspective or
head-pose distortion — hence the quality gate.

### Classification

Half-open intervals are used so values that land exactly on a boundary
(85.0, 90.0) are unambiguous. Classification always uses the **unrounded**
FI; rounding is display-only. Values outside every band are **never**
forced into a category.

| FI value | Category |
|---|---|
| `80.0 ≤ FI < 85.0` | Euryprosopic |
| `85.0 ≤ FI < 90.0` | Mesoprosopic |
| `90.0 ≤ FI < 95.0` | Leptoprosopic |
| otherwise | Outside the specified reference ranges |

---

## 2. The four landmarks

| Key | Symbol | Definition |
|---|---|---|
| `nasion` | N′ | Soft-tissue nasion — root of the nose, between the eyes |
| `menton` | Me′ | Soft-tissue menton — lowest point of the external chin contour |
| `zygion_right` | Zyᵣ | Right zygion — most-lateral anthropometric point of the right zygomatic region (subject's right) |
| `zygion_left` | Zyₗ | Left zygion — same definition on the left side |

**Do not substitute:**
- Pogonion or gnathion for menton.
- Generic cheek-mesh outline points for zygion.

The current MediaPipe baseline **approximates** these points from mesh
indices (168, 152, 234, 454). These are **candidates for visualisation
only** — they are *not* validated anthropometric labels. Manual
confirmation or the custom heatmap model is required before trusting any
measurement (see [limitations](#12-important-limitations)).

---

## 3. How the pipeline works

```
video / webcam
      │
      ▼
┌─────────────┐   sample at configurable fps, keep timestamps
│ Video reader │──────────────────────────────────────────────┐
└─────────────┘                                               │
      ▼                                                       │
┌─────────────┐   blur (Laplacian variance)                   │
│ Quality gate │── resolution ── head pose (yaw/pitch/roll) ──►│ reject
└─────────────┘                                               │ accept
      ▼                                                       │
┌─────────────┐   MediaPipe Face Landmarker (baseline)        │
│ Face + LM   │── 4 candidate points ── geometry validation ──►│ flag/reject
└─────────────┘                                               │ ok
      ▼                                                       │
┌─────────────┐   short-window median filter (jitter reduction)
│ Temporal    │
│ filter      │
└─────────────┘
      ▼
┌─────────────┐   H, W, FI per accepted frame
│ Measurement │
└─────────────┘
      ▼
┌─────────────┐   median FI across accepted frames
│ Aggregation │── classify (unrounded) ──► report + annotated preview
└─────────────┘
```

Key design rules (from the architecture doc):

- **Reject rather than guess.** Blurry, occluded, or excessively rotated
  frames are discarded with an explicit reason, never measured.
- **Median aggregation.** A single bad frame cannot drag the result; the
  median FI across accepted frames is robust to outliers.
- **Temporal filter ≠ correctness.** Smoothing reduces random jitter; it
  cannot fix a systematically wrong landmark.
- **Auditability.** Per-frame measurements, rejection reasons, and manual
  corrections are stored in SQLite.

---

## 4. Quick start

### Prerequisites

- Python 3.10+ (tested on 3.14)
- Node.js 18+ (for the web UI)
- A webcam (for the live demo) **or** a short near-frontal face video

### Install

```bash
# Backend
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

The MediaPipe model file `face_landmarker.task` must exist at
`backend/app/models/face_landmarker.task`. If it is missing, download it
(see [`backend/app/models/README.md`](backend/app/models/README.md) for
the exact URL and licence notes).

### Option A — live webcam demo (simplest)

```bash
cd backend
python webcam_demo.py
```

A window opens with your camera feed, landmark overlay, live pose
readout, and FI + category. Keys:

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `space` | Reset the measurement buffer |
| `d` | Toggle debug overlay |

> **Tip:** close the Windows Camera app (or any other app using the
> camera) first — otherwise OpenCV receives black frames and no face is
> detected.

### Option B — API server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Interactive docs: <http://127.0.0.1:8000/docs>

### Option C — web UI

```bash
# Terminal 1 — API
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Upload a video, review the FI report, and
drag landmark points to correct them.

---

## 5. Project structure

```
.
├── Research/
│   ├── AGENT_PLAN.md          # problem statement, goals, annotation plan, risks
│   └── ARCHITECTURE.md        # system design, data flow, module layout
├── backend/
│   ├── webcam_demo.py         # live OpenCV webcam demo (standalone)
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py            # FastAPI app + endpoints
│   │   ├── config.py          # thresholds, FI ranges, landmark indices
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── models/
│   │   │   ├── README.md      # model provenance & licence records
│   │   │   └── face_landmarker.task   # MediaPipe weights (downloaded)
│   │   └── services/
│   │       ├── video_reader.py        # OpenCV decode + frame sampling
│   │       ├── quality_gate.py        # blur / resolution / pose checks
│   │       ├── face_detector.py       # MediaPipe Face Landmarker baseline
│   │       ├── landmark_validation.py # geometry plausibility checks
│   │       ├── temporal_filter.py     # short-window median filter
│   │       ├── measurement.py         # H, W, FI, classifier (pure functions)
│   │       ├── report_builder.py      # per-frame → aggregate report
│   │       ├── storage.py             # SQLite persistence
│   │       ├── landmark_model.py      # PyTorch four-heatmap scaffold (Phase 3)
│   │       └── pipeline.py            # end-to-end orchestration
│   └── tests/                 # 54 unit + integration tests
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── api/               # typed client + types mirroring backend schemas
│       ├── components/
│       │   └── WebcamCapture.tsx   # getUserMedia + MediaRecorder
│       └── pages/
│           ├── UploadPage.tsx      # file upload
│           └── ResultPage.tsx      # FI report + drag-to-correct canvas
├── data/
│   └── README.md              # dataset provenance rules — never commit biometric data
└── docs/                      # (reserved)
```

---

## 6. API reference

All endpoints are served by FastAPI (default `http://127.0.0.1:8000`).

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status + model version |
| `POST` | `/api/v1/analyses` | Upload a video and run the pipeline |
| `GET` | `/api/v1/analyses` | List all analyses |
| `GET` | `/api/v1/analyses/{job_id}` | Full result for one analysis |
| `GET` | `/api/v1/analyses/{job_id}/frames` | Per-frame metadata + landmarks |
| `PATCH` | `/api/v1/analyses/{job_id}/landmarks/{frame_id}` | Submit manual corrections |
| `DELETE` | `/api/v1/analyses/{job_id}` | Delete stored result + artifacts |

### Upload example

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/analyses?config=%7B%22sample_fps%22%3A2%2C%22max_frames%22%3A60%7D" \
  -F "video=@face_video.mp4"
```

Accepted types: `.mp4 .mov .avi .mkv .webm` · max 500 MB.

Optional `config` query param (JSON):

```json
{
  "sample_fps": 2.0,
  "max_frames": 90,
  "min_laplacian_var": 40.0,
  "max_yaw_deg": 18.0,
  "max_pitch_deg": 18.0,
  "max_roll_deg": 15.0,
  "median_window": 5
}
```

### Correction example

```bash
curl -X PATCH "http://127.0.0.1:8000/api/v1/analyses/{job_id}/landmarks/{frame_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "frame_id": "f_000120",
    "landmarks": {
      "nasion":       {"x": 512, "y": 190, "source": "manual"},
      "menton":       {"x": 516, "y": 610, "source": "manual"},
      "zygion_right": {"x": 360, "y": 370, "source": "manual"},
      "zygion_left":  {"x": 670, "y": 372, "source": "manual"}
    }
  }'
```

The response includes the updated frame and the re-computed aggregate
FI / category. Corrections are logged in SQLite for audit.

---

## 7. Webcam demo

`backend/webcam_demo.py` is a standalone script that reuses the same
backend services as the API — no server required.

```bash
cd backend
python webcam_demo.py
```

**On screen you will see:**

- Mirrored live camera feed
- Face bounding box (green = accepted, orange = rejected this frame)
- Four coloured dots: N′ (red), Me′ (orange), Zy-r (blue), Zyₗ (purple)
- Yellow line = facial height · orange line = bizygomatic width
- HUD: FI value, category, accepted-frame count, live yaw/pitch/roll,
  blur score

**Why a frame might show REJECTED:**

| Reason | Meaning | What to do |
|---|---|---|
| `no_face_detected` | No face in frame | Move into view, improve lighting |
| `blur:N` | Laplacian variance too low | Improve focus / lighting |
| `yaw_out_of_range` | Head turned too far left/right | Face the camera more directly |
| `pitch_out_of_range` | Looking up/down too much | Level your head |
| `roll_out_of_range` | Head tilted too much | Straighten your head |
| `geometry_invalid` | Landmarks failed plausibility checks | Reposition; this should be rare |

Only frames that pass **all** gates enter the measurement buffer. The FI
displayed is the median-filtered value over recent accepted frames.

> **Common gotcha:** close the Windows Camera app before running the
> demo. An open camera app holds exclusive access and OpenCV receives
> black frames.

---

## 8. Web UI

The React frontend provides:

1. **Upload page** — choose a video file (mp4 / mov / avi / mkv / webm).
2. **Result page** —
   - Aggregate FI, category, accepted-frame count
   - Per-frame table with rejection reasons
   - Frame picker (accepted + flagged frames)
   - **Drag-to-correct canvas** — click and drag any landmark point,
     then "Save manual corrections" to recompute FI and category
   - Explicit warnings about baseline-landmark limitations

The frontend proxies `/api` and `/health` to `http://localhost:8000`
(see `frontend/vite.config.ts`).

---

## 9. Models

### MediaPipe Face Landmarker (baseline — active)

- **Role:** face detection, tracking, and 478-point mesh landmarks.
- **Use here:** four candidate points are selected from fixed mesh indices
  (`nasion=168`, `menton=152`, `zygion_right=234`, `zygion_left=454`).
- **Limitation:** these are **not** validated anthropometric labels. The
  mesh topology does not guarantee the exact targets required by the
  formula.
- **Licence:** MediaPipe source is Apache-2.0. Verify the exact `.task`
  asset's terms before redistribution.
- **Model file:** `backend/app/models/face_landmarker.task`
  (see [`backend/app/models/README.md`](backend/app/models/README.md)).

### Custom four-heatmap PyTorch model (scaffold — Phase 3)

- **File:** `backend/app/services/landmark_model.py`
- **Architecture:** small encoder–decoder CNN (`HeatmapNet`) that outputs
  four 128×128 heatmaps (one per landmark). Coordinates are extracted via
  softmax-weighted soft-argmax.
- **Status:** architecture verified (forward pass produces
  `(1, 4, 128, 128)`); **not yet integrated into the pipeline** and **no
  trained checkpoint exists**. Training requires an expert-annotated
  dataset — see [`data/README.md`](data/README.md) and AGENT_PLAN §8.
- **To activate:** train the model, save weights to
  `backend/app/models/heatmap_model.pt`, and wire `load_model()` /
  `predict()` into `pipeline.py` (currently the pipeline always uses the
  MediaPipe baseline).

---

## 10. Configuration

All tunable thresholds live in [`backend/app/config.py`](backend/app/config.py).

| Parameter | Default | Meaning |
|---|---|---|
| `DEFAULT_SAMPLE_FPS` | `2.0` | Frames per second analysed |
| `DEFAULT_MAX_FRAMES` | `90` | Hard cap on analysed frames |
| `DEFAULT_MIN_LAPLACIAN_VAR` | `40.0` | Blur gate (variance of Laplacian) |
| `DEFAULT_MAX_YAW_DEG` | `18.0` | Near-frontal yaw limit |
| `DEFAULT_MAX_PITCH_DEG` | `18.0` | Near-frontal pitch limit |
| `DEFAULT_MAX_ROLL_DEG` | `15.0` | Near-frontal roll limit |
| `DEFAULT_MEDIAN_WINDOW` | `5` | Temporal median filter length (odd) |
| `FI_RANGES` | see §1 | Half-open classification intervals |

The webcam demo (`webcam_demo.py`) uses **looser** gates (yaw/pitch ≤ 30°,
roll ≤ 25°, blur ≥ 15) so a typical laptop camera works interactively.
The API defaults are stricter.

---

## 11. Testing

```bash
cd backend
python -m pytest tests -q
```

**54 tests** covering:

- `test_measurement.py` — distances, H/W/FI, invalid widths, interval
  boundaries (85.0 → Mesoprosopic, not Euryprosopic), unrounded
  classification, median aggregation
- `test_validation_and_filter.py` — geometry validation (zero width,
  menton above nasion, width-too-small-vs-face), temporal median filter
- `test_report_builder.py` — aggregate only over accepted frames,
  outlier resistance, category-from-unrounded
- `test_api.py` — health, upload validation (wrong type / empty /
  bad config), full lifecycle on a synthetic clip (correctly returns
  FI=None when no face is present), 404s, DELETE

Frontend:

```bash
cd frontend
npm run typecheck   # tsc, zero errors
npm run build       # production build
```

---

## 12. Important limitations

Read these before trusting any output:

1. **Baseline landmarks are candidates, not ground truth.** MediaPipe
   mesh indices 168/152/234/454 approximate N′/Me′/Zyᵣ/Zyₗ for
   visualisation. Menton is not pogonion or gnathion. Zygion is not the
   arbitrary outer cheek-mesh point. Expert confirmation or the custom
   heatmap model is required.
2. **2D projection distortions.** Perspective and head pose change the
   apparent H/W ratio. The quality gate rejects clearly bad poses but
   cannot eliminate all distortion. Standardise capture distance and
   angle.
3. **Threshold sensitivity.** A small measurement error near a category
   boundary (85.0 or 90.0) can flip the label. Always report FI and
   quality, not category alone.
4. **Untrained custom model.** The PyTorch heatmap network exists as
   architecture only. It has never been trained and is not used at
   inference time.
5. **No clinical validation.** This prototype has not been validated
   against expert anthropometric measurements. Do not use it for
   diagnosis, identification, or forensic conclusions.

---

## 13. Privacy

- **Local-first.** All inference runs on your machine. No cloud vision
  API is involved.
- **Raw video is deleted immediately** after analysis (API deletes the
  upload workspace as soon as the result is stored).
- **No biometric data is committed.** `data/workspace/` is git-ignored.
  See [`data/README.md`](data/README.md).
- **Consent.** Obtain informed consent before recording or analysing
  anyone's face. Provide deletion controls if you deploy this beyond a
  personal prototype.

---

## 14. Roadmap

Based on AGENT_PLAN §10:

| Phase | Scope | Status |
|---|---|---|
| **1** | Formula, interval boundaries, unit tests | ✅ Done |
| **2** | Video pipeline, quality gate, MediaPipe baseline, overlay UI | ✅ MVP done |
| **3** | Expert annotation → train custom four-heatmap model | ⬜ Scaffold only — blocked on annotated dataset |
| **4** | Confidence gating, temporal filtering, formal evaluation | ◐ Median filter done; evaluation pending data |
| **5** | Result report, correction workflow, documentation | ◐ Correction + report done; docs in progress |

### Known gaps (honest list)

- Webcam capture UI component exists (`WebcamCapture.tsx`) but is not yet
  wired into the main upload flow — use `webcam_demo.py` for live capture.
- Heatmap model is not integrated into `pipeline.py` (always uses
  MediaPipe baseline).
- Annotated preview in the web UI draws landmarks on a blank canvas, not
  on the actual video frame (backend does not return frame images).
- No formal evaluation notebook or expert-annotated dataset yet.

---

## 15. Licences

| Component | Licence | Notes |
|---|---|---|
| MediaPipe | Apache-2.0 | Verify the exact `.task` asset terms before redistribution |
| OpenCV | Apache-2.0 | Check the distribution details for your build |
| FastAPI | MIT | |
| PyTorch | BSD-style | Model weights have their own provenance |
| React / Vite | MIT | |
| This project's code | — | Add your preferred licence before publishing |

References:
[MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) ·
[MediaPipe repo](https://github.com/google-ai-edge/mediapipe) ·
[WFLW](https://github.com/wywu/WFLW) ·
[300-VW](https://ibug.doc.ic.ac.uk/resources/300-VW/)

---

**Disclaimer:** This is a technical prototype for research and education.
It is not a medical device, not clinical software, and not a substitute
for expert anthropometric assessment.

# Architecture --- Video-Based Facial Index Classification

## 1. Architecture objectives

The system accepts live webcam video or prerecorded video, estimates
four anthropometric landmarks on suitable frames, calculates the Facial
Index (FI), and classifies it using configured ranges.

Design principles: - Local-first inference; no paid cloud vision API. -
Keep face detection, landmark estimation, measurement, and
classification as separate modules. - Never treat generic face-mesh
points as exact anthropometric labels without validation. - Preserve
timestamps and auditability. - Reject unsuitable frames rather than
producing a confident-looking result from poor input. - Make manual
correction possible.

## 2. High-level architecture

``` mermaid
flowchart TD
    U[React + TypeScript UI] -->|Webcam or video upload| API[FastAPI service]
    API --> V[OpenCV video decoder and frame sampler]
    V --> Q[Frame quality gate]
    Q -->|Reject| R[Quality warnings / recapture]
    Q -->|Accept candidate| FD[Face detection and crop]
    FD --> MP[MediaPipe Face Landmarker baseline]
    FD --> HM[Custom PyTorch heatmap model]
    MP --> SEL[Landmark selection / mapping]
    HM --> SEL
    SEL --> VAL[Confidence and geometry validation]
    VAL -->|Low confidence| REV[Manual review / correction UI]
    REV --> VAL
    VAL -->|Valid frame| TEMP[Temporal association and jitter filtering]
    TEMP --> MEAS[Measurement engine]
    MEAS --> AGG[Robust aggregation across accepted frames]
    AGG --> CLS[FI range classifier]
    CLS --> REP[Result report and annotated preview]
    REP --> U
    API --> DB[(SQLite metadata / results)]
```

**Implementation note:** During the first baseline phase, MediaPipe may
be used to visualize generic landmarks, but the four target points must
be manually confirmed or produced by the custom model. Do not assume a
mesh index is the correct zygion or menton just because it is nearby.

## 3. Recommended technology stack

  -----------------------------------------------------------------------
  Layer                   Technology              Responsibility
  ----------------------- ----------------------- -----------------------
  Frontend                React, TypeScript, Vite Webcam/upload, video
                                                  progress, point
                                                  overlay, correction
                                                  controls, FI report

  Backend API             Python, FastAPI,        Validate uploads,
                          Uvicorn                 create analysis jobs,
                                                  return results

  Video processing        OpenCV                  Decode video, sample
                                                  frames, timestamps,
                                                  blur checks, overlays

  Baseline landmarks      MediaPipe Face          Generic face landmarks
                          Landmarker              and tracking; candidate
                                                  visualization

  Target landmarks        PyTorch                 Predict N′, Me′, Zyᵣ,
                          heatmap-regression      Zyₗ heatmaps
                          model                   

  Model backbone          A small pretrained      Feature extraction;
                          backbone with a         exact checkpoint
                          verified permissive     license must be
                          license, or a small     recorded
                          custom CNN              

  Math and metrics        NumPy; scikit-learn for Distances, aggregation,
                          offline evaluation      FI evaluation
                          metrics                 

  Annotation              Self-hosted CVAT/Label  Human-labelled
                          Studio Community        training/validation
                          Edition or a simple     points
                          point-click tool        

  Tests                   pytest                  Formula, validation,
                                                  API, and pipeline tests

  Prototype storage       SQLite                  Analysis metadata and
                                                  results; no raw video
                                                  retention by default

  Future multi-user       PostgreSQL (optional)   Users, jobs, results,
  storage                                         audit metadata

  Packaging               venv/uv or pip; Docker  Reproducible
                          optional                environment and local
                                                  deployment
  -----------------------------------------------------------------------

### Models: baseline and target

**Baseline --- MediaPipe Face Landmarker** - Purpose: face
detection/tracking and generic mesh landmarks. - Use: initialize the
face crop, visualize a baseline, inspect temporal tracking. -
Limitation: its generic landmark topology does not guarantee the exact
anthropometric targets needed for the formula. - License: MediaPipe
source is Apache-2.0. Verify the exact downloaded task/model asset's
terms before bundling or redistributing it. - Official docs:
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

**Target --- custom four-channel heatmap model (PyTorch)** - Input:
normalized face crop. - Output: four heatmaps, one each for N′, Me′,
Zyᵣ, and Zyₗ. - Training labels: expert-annotated images using a fixed
landmark protocol. - Coordinate extraction: heatmap peak with optional
local refinement. - Training: fine-tune a pretrained backbone only after
confirming the backbone/checkpoint license, or train a small model from
scratch. - The custom model's rights depend on the code, starting
checkpoint, training data, and resulting weights; document all of them.

**Optional detector --- Ultralytics YOLO11/YOLO26** - Not required in
the initial architecture; MediaPipe is the simpler default for
face-focused processing. - YOLO can be considered for face detection
only if a suitable face-trained checkpoint is separately identified and
license-checked. General YOLO object-detection weights do not
automatically detect faces as a dedicated class. - Ultralytics
code/models use AGPL-3.0 with a paid Enterprise alternative. AGPL costs
nothing but imposes obligations, including source-sharing requirements
in covered circumstances. Do not integrate it into a closed/private
deployment without confirming license compatibility. - Official
licensing: https://www.ultralytics.com/license

No paid hosted inference service is required. Software being free of
charge does not eliminate license, dataset, privacy, or model-weight
conditions.

## 4. Landmark contract

All coordinates are represented in original-frame pixel coordinates
after mapping back from any crop/resize transform.

  -----------------------------------------------------------------------
  Key                     Anatomical target       Use
  ----------------------- ----------------------- -----------------------
  `nasion` / N′           Soft-tissue nasion:     Top of facial height
                          root of the nose        
                          between the eyes        

  `menton` / Me′          Soft-tissue menton:     Bottom of facial height
                          lowest point of the     
                          external chin contour   

  `zygion_right` / Zyᵣ    Right zygion under the  One endpoint of
                          agreed anthropometric   bizygomatic width
                          definition              

  `zygion_left` / Zyₗ     Left zygion under the   Other endpoint of
                          same definition         bizygomatic width
  -----------------------------------------------------------------------

**Do not substitute:** - Pogonion for menton. - Gnathion for menton. -
The generic left/right cheek outline or arbitrary mesh vertices for
zygion.

The supplied Dynasmile paper lists soft-tissue nasion and smile-analysis
landmarks, including soft-tissue pogonion; it does not define the
complete four-point set for this Facial Index. It is a useful
video-workflow reference, not a substitute for the four-point annotation
protocol.

## 5. Measurement and classification module

Given landmark coordinates in pixels:

``` text
distance(A, B) = sqrt((Ax - Bx)^2 + (Ay - By)^2)

H = distance(N′, Me′)
W = distance(Zyᵣ, Zyₗ)

FI = (H / W) * 100
```

Validation: - If any required point is missing or invalid, do not
calculate FI. - If `W <= 0`, reject the frame. - Reject/flag geometry
that fails validation checks calibrated from labelled data. - Keep full
precision for classification; round only for display. - Since H and W
are both in pixels, the units cancel in the ratio, but camera
perspective and head pose still affect the result.

Initial configured category ranges:

  Rule                  Category
  --------------------- --------------------------
  `80.0 <= FI < 85.0`   Euryprosopic
  `85.0 <= FI < 90.0`   Mesoprosopic
  `90.0 <= FI < 95.0`   Leptoprosopic
  Otherwise             Outside reference ranges

These half-open intervals remove ambiguity at 85 and 90 while preserving
the supplied one-decimal bands. Confirm the intervals against the
authoritative reference used by the project. Do not force values outside
the specified ranges into a category.

## 6. Runtime data flow

### A. Input

1.  User selects webcam or video file.
2.  Frontend checks file type/size and submits the video.
3.  Backend assigns a job ID and stores temporary video only for the
    duration needed to process it.

### B. Frame sampling and quality checks

1.  OpenCV decodes frames and preserves timestamps.
2.  Sample frames at a configurable rate to limit compute.
3.  Check resolution, blur, face visibility, and approximate head pose.
4.  Reject or flag frames with excessive yaw/pitch/roll, occlusion, or
    blur.

### C. Face crop and landmarks

1.  Detect/track the face and derive a consistent crop.
2.  Run MediaPipe for baseline visualization/tracking.
3.  Run the custom heatmap model for the four task-specific points.
4.  Map heatmap coordinates back to the original frame.
5.  Attach per-point quality/confidence metadata.

### D. Validation and temporal stabilization

1.  Check that all four points are available.
2.  Check positive bizygomatic width and plausible geometry.
3.  Associate the same face across accepted frames.
4.  Apply a short-window median or another validated temporal filter to
    reduce random jitter.
5.  Send uncertain points to the correction UI.
6.  Do not let smoothing silently override an anatomically wrong
    prediction.

### E. Measurement and report

1.  Calculate H, W, and FI for each accepted frame.
2.  Aggregate using the median FI across valid near-frontal frames.
3.  Retain the per-frame measurements for debugging and audit.
4.  Assign the category from the unrounded aggregate FI.
5.  Return the result, point overlay, accepted timestamps, quality
    notes, and model version.

## 7. Backend modules and suggested API

Suggested modules:

``` text
backend/
  app/
    main.py                 # FastAPI app and routes
    schemas.py              # Request/response models
    services/
      video_reader.py       # OpenCV decode and frame sampling
      quality_gate.py       # Blur, visibility, pose/quality checks
      face_detector.py      # MediaPipe baseline / optional detector
      landmark_model.py     # PyTorch heatmap inference
      landmark_validation.py
      temporal_filter.py
      measurement.py        # H, W, FI and category rules
      report_builder.py
      storage.py
    models/
      README.md             # Model provenance, version, and license records
  tests/
```

Suggested API: - `GET /health` --- service status. -
`POST /api/v1/analyses` --- submit a video or start a supported local
capture workflow. - `GET /api/v1/analyses/{job_id}` --- retrieve job
state and summary. - `GET /api/v1/analyses/{job_id}/frames` --- retrieve
accepted-frame metadata and landmark overlays. -
`PATCH /api/v1/analyses/{job_id}/landmarks/{frame_id}` --- submit a
user's corrected points. - `DELETE /api/v1/analyses/{job_id}` --- delete
stored result/video artifacts according to retention policy.

For the first MVP, synchronous processing on short clips is acceptable.
Move to a background worker only if processing time requires it.

## 8. Data model

Minimal metadata tables/files:

-   `analysis`: job ID, creation time, status, model version,
    configuration, aggregate FI, category, quality summary.
-   `frame_result`: analysis ID, timestamp, frame quality status, H, W,
    FI, accepted/rejected reason.
-   `landmark`: frame result ID, point name, x, y,
    confidence/uncertainty, source (model/manual).
-   `correction_log`: old/new coordinates, reviewer, timestamp, reason.
-   `model_registry`: model version, architecture, checkpoint hash,
    training dataset/version, license/provenance.

Avoid saving raw video by default. If retention is needed, obtain
consent, define access controls, and provide deletion.

## 9. Training and evaluation architecture

``` mermaid
flowchart LR
    D[Consent and licensed images/videos] --> A[Expert annotation]
    A --> QA[Annotation QA / adjudication]
    QA --> SPLIT[Subject-disjoint train/validation/test split]
    SPLIT --> TRAIN[PyTorch heatmap training]
    TRAIN --> VAL[Offline validation and calibration]
    VAL --> REG[Versioned model + provenance record]
    REG --> INF[Runtime inference service]
    VAL --> REPORT[Evaluation report]
```

Training requirements: - Define each landmark in a written annotation
guide with examples and edge cases. - Have two annotators label a pilot
subset independently; adjudicate disagreements. - Include
unobservable/invalid labels rather than guessing. - Split by subject,
never randomly by frame alone. - Keep the held-out test set separate
from model tuning. - Public datasets such as WFLW and 300-VW can be
useful for initialization or temporal tests, but their existing
landmarks are not ground truth for the four exact anthropometric
points. - Use a custom labelled dataset for the actual task.

Evaluation metrics: - Per-point Euclidean error and normalized mean
error. - Percentage of points within an expert-agreed tolerance. - FI
absolute error against expert reference. - Category accuracy, macro-F1,
and confusion matrix. - Temporal variance and repeatability. - Video
acceptance rate, rejection reasons, and manual correction rate.

## 10. Suggested repository layout

``` text
facial-index-video/
  README.md
  AGENT_PLAN.md
  ARCHITECTURE.md
  frontend/
    src/
      components/
      pages/
      api/
  backend/
    app/
    tests/
    models/
      README.md
  data/
    README.md              # Dataset provenance; do not commit sensitive videos
    annotations/
  notebooks/
    evaluation.ipynb
  docs/
    landmark-annotation-guide.md
    model-card.md
    privacy-and-retention.md
  requirements.txt
  pyproject.toml
  .env.example
  .gitignore
```

## 11. Risks and mitigations

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Wrong anatomical point despite      Expert-defined labels, independent
  stable tracking                     annotation, adjudication, point
                                      overlays

  Head pose distorts 2D width         Near-frontal capture guidance, pose
                                      gate, reject unsuitable frames

  Frame-to-frame jitter               Face association and short-window
                                      filter; report raw and filtered
                                      values

  Model bias across subjects          Diverse subject-disjoint evaluation
                                      and subgroup error reporting

  Category flips near thresholds      Preserve full precision, report FI
                                      and uncertainty, validate boundary
                                      behavior

  Dataset or checkpoint license       Record provenance and license per
  mismatch                            asset; verify before distribution

  Sensitive face-video data           Local processing, consent, minimal
                                      retention, deletion controls
  -----------------------------------------------------------------------

## 12. Implementation order

1.  Implement formula, interval boundaries, and unit tests.
2.  Build manual four-point annotation and overlay UI.
3.  Build video reader, frame sampling, and quality gate.
4.  Add MediaPipe baseline for face localization and visualization.
5.  Collect expert-annotated task-specific data.
6.  Train the custom heatmap model and compare against manual labels.
7.  Add temporal filtering, correction flow, and report generation.
8.  Evaluate on held-out subjects and document limitations/licenses.

## 13. License and reference links

-   MediaPipe Face Landmarker:
    https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
-   MediaPipe source repository:
    https://github.com/google-ai-edge/mediapipe
-   Ultralytics YOLO11: https://docs.ultralytics.com/models/yolo11/
-   Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26
-   Ultralytics licensing: https://www.ultralytics.com/license
-   WFLW: https://github.com/wywu/WFLW
-   300-VW: https://ibug.doc.ic.ac.uk/resources/300-VW/
-   Dynasmile paper code: https://github.com/dentistfrankchen/dynasmile

This is a technical architecture proposal, not a legal opinion or a
claim of clinical validity. Verify licenses for exact model assets,
checkpoints, dependencies, and datasets before distribution.

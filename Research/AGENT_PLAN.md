# Agent Plan --- Video-Based Facial Index Classification

## 1. Project summary

Build a local-first computer-vision application that accepts a live
webcam stream or a prerecorded video, selects suitable near-frontal
frames, estimates four specifically defined facial landmarks, calculates
the Facial Index (FI), and reports the corresponding facial-index
category.

**The system is a measurement and classification prototype, not a
medical diagnostic system.** Its outputs depend on camera perspective,
pose, image quality, the chosen anthropometric definitions, and the
accuracy of the landmark annotations. The method must be validated
against an expert reference before clinical or forensic use.

## 2. Problem statement

Manual facial-index measurement is time-consuming and may vary between
operators. An automated system can make the workflow more repeatable,
but generic facial-landmark models do not necessarily provide the exact
anthropometric points required by the formula.

The key challenge is therefore not simply detecting a face. It is
consistently locating these four targets:

1.  **N′ --- soft-tissue nasion:** the surface point at the root of the
    nose, between the eyes.
2.  **Me′ --- soft-tissue menton:** the lowest point on the external
    soft-tissue chin contour.
3.  **Zyᵣ --- right zygion:** the designated most-lateral anthropometric
    point of the subject's right zygomatic region.
4.  **Zyₗ --- left zygion:** the corresponding point on the subject's
    left zygomatic region.

> **Important distinction:** Menton is not the same as pogonion or
> gnathion. Zygion is not automatically the outermost cheek-mesh point
> from a generic face model. The annotation protocol must define the
> intended landmark and be reviewed by someone with appropriate
> anthropometric expertise.

## 3. Goal and success criteria

### Main goal

Given a suitable frontal video, return: - The accepted frame(s) used for
measurement. - Four landmark coordinates and their confidence/quality
status. - Facial height, bizygomatic width, and FI. - A category only
when the FI is within a defined reference range. - An annotated preview
that allows the user to inspect or correct the points.

### Engineering goals

-   Process video by analysing selected frames rather than assuming
    every frame is usable.
-   Reject or flag blurred, occluded, or excessively rotated frames.
-   Reduce random frame-to-frame jitter without hiding systematic
    landmark errors.
-   Keep the complete pipeline runnable locally without a paid API or
    paid inference service.
-   Save a reproducible record of the model version, thresholds,
    accepted frames, and measurement result.

### Initial acceptance targets (to be agreed after a pilot)

These are proposed evaluation targets, not claims of current
performance: - Report landmark error in pixels and normalized by face
size. - Measure FI error against expert-labelled reference frames. -
Measure category accuracy, macro-F1, and confusion matrix on a
subject-disjoint test set. - Measure repeatability across accepted
frames and repeated recordings. - Track the percentage of videos
rejected for poor quality. - Set tolerances with an expert before
declaring the system successful; pay special attention to samples close
to category boundaries.

## 4. Formula and category rules

Let the 2D image coordinates be `(x, y)`.

Euclidean distance:

`d(A, B) = sqrt((x_A - x_B)^2 + (y_A - y_B)^2)`

Facial height:

`H = d(N′, Me′)`

Bizygomatic width:

`W = d(Zyᵣ, Zyₗ)`

Facial Index:

`FI = (H / W) × 100`

Because both measurements use the same image-coordinate units, pixels
cancel in the ratio. This does **not** remove perspective or head-pose
distortion.

Use the following explicit half-open intervals to avoid ambiguous
boundaries in the supplied one-decimal ranges:

  FI value             Category
  -------------------- ----------------------------------------
  `80.0 ≤ FI < 85.0`   Euryprosopic
  `85.0 ≤ FI < 90.0`   Mesoprosopic
  `90.0 ≤ FI < 95.0`   Leptoprosopic
  Otherwise            Outside the specified reference ranges

Classify using the **unrounded** FI; round only for display. These
ranges should be confirmed against the project's authoritative reference
before use. Do not force a category for values outside the listed
ranges.

## 5. Recommended open-source technology stack

All inference should run locally in the initial version. No paid hosted
vision API is required.

  -----------------------------------------------------------------------------------
  Component         Recommended          Role                 License / cost note
                    technology                                
  ----------------- -------------------- -------------------- -----------------------
  Frontend          React + TypeScript + Video upload/webcam  Open-source; React is
                    Vite                 UI, annotated        MIT-licensed
                                         preview, result      
                                         report               

  API               Python + FastAPI +   Upload and analysis  Open-source; FastAPI is
                    Uvicorn              endpoints            MIT-licensed

  Video / image     OpenCV               Read video, sample   Open-source; verify
  handling                               frames, blur/pose    distribution details
                                         preprocessing,       for the chosen build
                                         overlay landmarks    

  Baseline face     MediaPipe Face       Face                 MediaPipe source is
  landmarks         Landmarker           detection/tracking   Apache-2.0; verify the
                                         and generic          exact model asset's
                                         face-mesh landmarks  terms before
                                         for initialization   redistributing it
                                         and comparison       

  Custom landmark   PyTorch              Predict separate     Open-source framework;
  model             heatmap-regression   spatial heatmaps for model weights depend on
                    model                N′, Me′, Zyᵣ, Zyₗ    training data and model
                                                              provenance

  Numerical         NumPy                Distances, FI,       Open-source
  calculations                           aggregation, metrics 

  Annotation        CVAT or Label Studio Expert annotation    Use the
                    Community Edition,   and review           self-hosted/community
                    or a small custom                         option; verify the
                    point-click tool                          selected version and
                                                              license

  Testing           pytest               Unit and API tests   Open-source

  Optional storage  SQLite for           Metadata and result  Open-source; not needed
                    prototype;           history              for the first local
                    PostgreSQL if                             demo
                    multi-user storage                        
                    is needed                                 
  -----------------------------------------------------------------------------------

### Model selection: what to use and why

**Recommended first baseline: MediaPipe Face Landmarker.** It provides
generic face landmarks and is useful for face tracking, initial point
visualization, and comparing frame-to-frame movement. Its landmarks are
not guaranteed to equal the four required anthropometric targets. Do not
simply choose the nearest cheek mesh vertices and call them zygion
without expert validation.

**Recommended target model: a custom heatmap-based keypoint detector in
PyTorch.** Use a pretrained, permissively licensed backbone if its exact
weight license is suitable, then fine-tune on images annotated for the
four required targets. The model outputs four heatmaps, one per target.
A heatmap represents the predicted location distribution for a point;
the peak or a local weighted average gives its coordinate. Heatmap
uncertainty can support rejection or manual review, but does not by
itself prevent systematic anatomical misplacement.

**Optional face detector: YOLO11 or YOLO26.** These are not required if
MediaPipe handles face detection adequately. Ultralytics publishes the
software and models under AGPL-3.0 with an Enterprise alternative. AGPL
is free of charge but has source-sharing obligations; it is not the
right default for a private/proprietary application unless the licensing
terms are acceptable. For this project, avoid making YOLO a mandatory
dependency. If it is evaluated, check the current official license and
obligations before integrating it.

**Not recommended as a drop-in solution:** a generic pose model such as
a body pose estimator (e.g., models intended for RULA/REBA posture
scoring). Those are designed for body posture/ergonomics and do not
define the required facial anthropometric landmarks.

## 6. Why heatmaps?

A direct regression model predicts four `(x, y)` coordinate pairs. A
heatmap model instead predicts one 2D probability-like map per landmark.
This gives a spatial representation that can be visualized and reviewed.

Proposed training target: - Input: aligned/cropped face image. - Output:
four heatmaps at a reduced spatial resolution. - Ground truth: Gaussian
heatmap centred at each expert-labelled point. - Loss: mean squared
error or a suitable heatmap loss. - Coordinate extraction: peak
location, optionally refined with a local weighted average. -
Confidence: peak strength/shape and validation-calibrated thresholds; do
not interpret raw peak values as calibrated probabilities without
calibration.

Use the heatmap model only after a consistent annotation guide and a
small pilot dataset exist.

## 7. End-to-end workflow

1.  **Input:** user uploads a video or starts a webcam capture.
2.  **Decode and sample:** OpenCV samples frames at a configurable rate.
    Keep original timestamps.
3.  **Quality gate:** detect face; check image resolution, blur,
    occlusion, and approximate head pose. Accept near-frontal frames
    only.
4.  **Face crop/alignment:** normalize scale and crop consistently.
    Record transformations so predictions can be mapped back to
    original-frame coordinates.
5.  **Landmark estimation:** run MediaPipe as the baseline; run the
    custom four-heatmap model when available.
6.  **Landmark validation:** check that all four points exist, width is
    positive, points are plausible relative to the face, and confidence
    is adequate. Flag low-confidence or inconsistent frames for review.
7.  **Temporal stabilization:** associate the same face across frames
    and apply a short-window median or other validated filter to reduce
    random jitter. Filtering cannot correct a systematically wrong
    point.
8.  **Measurement:** calculate H, W, and FI per accepted frame.
9.  **Aggregation:** use a robust statistic (initially the median FI
    across accepted, high-quality frames) and retain per-frame
    measurements for audit. Do not average bad poses into good
    measurements.
10. **Classification:** apply the fixed interval rules to the unrounded
    aggregate FI.
11. **Review and report:** show the selected frame, four labelled
    points, measurements, category, quality warnings, and an option to
    manually correct points.
12. **Evaluation log:** store model/version, accepted frame timestamps,
    corrections, and evaluation metadata when the user has consented.

## 8. Annotation plan

### Annotation rules

-   Annotate the four points on frontal, sufficiently sharp frames.
-   Use a written guide with example images and exclusion cases.
-   Label the subject's right and left consistently (not
    screen-left/screen-right).
-   Have two trained annotators label a pilot subset independently;
    adjudicate disagreements with an expert.
-   Include an `unobservable` / `do_not_use` status rather than
    inventing a point when a landmark is hidden or ambiguous.
-   Save coordinates in original-image pixels and document any
    crop/resize transform.
-   Split train/validation/test by **subject**, not by frames, to avoid
    leakage from the same person across splits.

### Suggested annotation schema

``` json
{
  "image_id": "subject_001_frame_000120",
  "subject_id": "subject_001",
  "timestamp_ms": 4000,
  "nasion": [512, 190],
  "menton": [516, 610],
  "zygion_right": [360, 370],
  "zygion_left": [670, 372],
  "quality_status": "accepted",
  "annotator_id": "annotator_01",
  "review_status": "pending"
}
```

The coordinates above are illustrative placeholders, not real
measurements. For a dataset, use a structured format such as JSONL or
COCO-style keypoints with a documented mapping.

### Dataset strategy

-   Start with a small, expert-labelled pilot set to validate the
    definitions and UI.
-   Public facial-landmark datasets such as WFLW or 300-VW may help
    initialize a model or evaluate general face tracking, but their
    existing point labels should **not** be treated as ground-truth
    labels for this exact four-point task.
-   Only use datasets after checking their access conditions, consent,
    data license, and permitted use.
-   The supplied Dynasmile paper is useful as a video-processing
    workflow reference. Its listed landmarks focus on smile analysis and
    do not supply the exact four-point Facial Index annotation set; its
    soft-tissue nasion definition can be a starting reference, not proof
    that the full target set is covered.

## 9. Evaluation plan

Evaluate on held-out subjects and report:

-   **Per-landmark error:** Euclidean pixel error and error normalized
    by face-box size.
-   **Landmark threshold accuracy:** percentage of points within an
    expert-agreed tolerance.
-   **FI error:** absolute difference between model FI and
    expert-reference FI.
-   **Category metrics:** accuracy, macro-F1, and confusion matrix.
-   **Temporal stability:** variation in point coordinates and FI over
    accepted frames.
-   **Repeatability:** repeated recordings of the same subject under the
    same capture setup.
-   **Coverage:** accepted/rejected video rate and reasons for
    rejection.
-   **Manual correction rate:** how often an operator must reposition a
    point.

Do not use temporal smoothness alone as evidence of anatomical accuracy.

## 10. Scope and milestones

### Phase 1 --- Formula and annotation UI

-   Implement four-point manual annotation and FI calculation.
-   Add unit tests for distances, interval boundaries, and invalid
    widths.
-   Review the landmark guide with an expert.

### Phase 2 --- Video pipeline and baseline

-   Add video decoding, frame sampling, quality checks, MediaPipe
    baseline landmarks, and annotated previews.
-   Test on a small set of consented videos.

### Phase 3 --- Custom heatmap model

-   Annotate a subject-diverse dataset.
-   Train and validate the four-heatmap detector.
-   Compare predictions with expert annotations.

### Phase 4 --- Stabilization and evaluation

-   Add confidence gating and temporal filtering.
-   Evaluate landmark error, FI error, categories, and repeatability on
    held-out subjects.

### Phase 5 --- Demo and documentation

-   Build the result report and correction workflow.
-   Document limitations, licenses, dataset provenance, and model
    versions.

## 11. Risks and safeguards

-   **Pose/perspective:** a frontal 2D projection can change apparent
    width; reject excessive yaw/pitch and standardize capture.
-   **Ambiguous zygion:** external cheek contour may not identify the
    anthropometric zygion reliably; require an agreed definition and
    expert review.
-   **Soft-tissue variation:** lighting, expression, makeup, facial
    hair, and occlusion may affect visible contours.
-   **Bias:** validate across a diverse subject set and report subgroup
    performance where ethically and statistically appropriate.
-   **Threshold sensitivity:** a small measurement error near a category
    boundary can change the label; report FI and quality, not category
    alone.
-   **Privacy:** process locally by default, request consent, minimize
    retained video, and provide deletion controls.
-   **Licensing:** no paid API is required, but "free" does not mean "no
    conditions." Record the license for each dependency, model weight,
    and dataset.

## 12. Official model and license references

-   MediaPipe Face Landmarker:
    https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
-   MediaPipe repository/license:
    https://github.com/google-ai-edge/mediapipe
-   Ultralytics YOLO11: https://docs.ultralytics.com/models/yolo11/
-   Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26
-   Ultralytics license: https://www.ultralytics.com/license
-   WFLW dataset repository: https://github.com/wywu/WFLW
-   300-VW dataset: https://ibug.doc.ic.ac.uk/resources/300-VW/
-   Dynasmile paper repository:
    https://github.com/dentistfrankchen/dynasmile

**License note:** These references are a starting point, not legal
advice. Before redistribution or deployment, verify the current license
for the exact code version, pretrained checkpoint/model asset, and
dataset. No hosted paid service is part of the proposed baseline.

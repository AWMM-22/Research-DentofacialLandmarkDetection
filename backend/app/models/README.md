# Model provenance and license records

Per AGENT_PLAN.md section 11 and ARCHITECTURE.md section 3, every model
asset used at runtime must have its license and provenance recorded here
before distribution or deployment.

## face_landmarker.task

- **Asset**: MediaPipe Face Landmarker, float16, version 1
- **Source URL**: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- **Downloaded**: 2026-10-14
- **Software license**: MediaPipe source is Apache-2.0
  (https://github.com/google-ai-edge/mediapipe)
- **Asset terms**: verify the exact downloaded `.task` file's terms before
  redistributing or bundling it with a product.
- **Use in this project**: baseline face detection/tracking and generic
  mesh landmarks for visualisation and initialisation ONLY. The four
  anthropometric targets (N', Me', Zy_r, Zy_l) are approximated from mesh
  indices and are **not** validated anthropometric labels
  (ARCHITECTURE.md section 4).

## heatmap_model.pt (not present)

- **Intended asset**: custom four-channel heatmap CNN (PyTorch), scaffold
  in `backend/app/services/landmark_model.py`.
- **Status**: no trained checkpoint exists yet. Training requires an
  expert-annotated dataset (AGENT_PLAN.md section 8).
- **License/provenance**: to be recorded when a checkpoint is trained —
  must document code license, any pretrained backbone and its weight
  license, training data provenance/consent, and the resulting weights.

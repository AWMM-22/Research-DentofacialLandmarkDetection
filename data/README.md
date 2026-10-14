# Dataset provenance

Do **not** commit videos, face images, or any personally identifiable
biometric data to this repository (ARCHITECTURE.md section 8, AGENT_PLAN
section 11).

## Layout

- `annotations/` — expert-labelled keypoints (JSONL or COCO-style). Only
  commit annotations that are covered by explicit participant consent and
  a documented data license.
- `workspace/` — runtime job artifacts (git-ignored). Raw uploaded video is
  deleted immediately after processing by the API.

## Annotation schema

See AGENT_PLAN.md section 8. Minimum required fields per record:

```json
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

Split train/validation/test by **subject**, never by frame, to avoid
leakage (AGENT_PLAN.md section 8).

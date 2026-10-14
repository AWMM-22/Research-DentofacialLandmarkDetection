"""FastAPI application: Facial Index video analysis service.

Implements ARCHITECTURE.md section 7 endpoints:

  GET    /health
  POST   /api/v1/analyses
  GET    /api/v1/analyses
  GET    /api/v1/analyses/{job_id}
  GET    /api/v1/analyses/{job_id}/frames
  PATCH  /api/v1/analyses/{job_id}/landmarks/{frame_id}
  DELETE /api/v1/analyses/{job_id}

MVP note (ARCHITECTURE.md section 7): synchronous processing on short
clips is acceptable. Uploads are validated for type/size before analysis.
"""

from __future__ import annotations

import json
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import MODEL_VERSION, WORKSPACE_DIR
from .schemas import (
    AggregateResult,
    AnalysisResult,
    AnalysisStatus,
    CorrectionIn,
    CorrectionOut,
    FrameQuality,
    FrameResult,
    HealthResponse,
    LandmarksModel,
    ModelConfig,
    ModelInfo,
    QualitySummary,
)
from .services import face_detector, pipeline, report_builder, storage
from .services.storage import init_db, workspace_for

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("facial-index")

DB_PATH = WORKSPACE_DIR / "facial_index.sqlite3"
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    init_db(DB_PATH)
    yield


app = FastAPI(
    title="Facial Index Video Analysis API",
    version="0.1.0",
    description=(
        "Local-first prototype: estimates four facial landmarks from video, "
        "computes the Facial Index, and classifies it. Not a medical device."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=face_detector.model_loaded(),
        model_version=MODEL_VERSION,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _result_from_row(row: dict) -> AnalysisResult:
    result = json.loads(row["result_json"]) if row.get("result_json") else {}
    frames = [FrameResult(**f) for f in result.get("frames", [])]
    return AnalysisResult(
        job_id=row["id"],
        status=AnalysisStatus(row["status"]),
        created_at=row["created_at"],
        completed_at=row.get("completed_at"),
        filename=row.get("filename"),
        model=ModelInfo(
            **result.get(
                "model",
                {"name": "unknown", "version": "unknown", "note": ""},
            )
        ),
        quality=QualitySummary(**result.get("quality", {})),
        aggregate=AggregateResult(
            **result.get(
                "aggregate",
                {"category": "Outside the specified reference ranges"},
            )
        ),
        representative_frame_id=result.get("representative_frame_id"),
        frames=frames,
        warnings=result.get("warnings", []),
    )


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
@app.post("/api/v1/analyses", response_model=AnalysisResult, status_code=201)
async def create_analysis(
    video: UploadFile = File(...),
    config: Optional[str] = Query(
        None,
        description=(
            "Optional JSON of ModelConfig overrides, e.g. "
            '{"sample_fps": 1.0, "max_frames": 30}'
        ),
    ),
) -> AnalysisResult:
    if video.filename is None:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(video.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video type {suffix!r}; "
                f"allowed: {sorted(ALLOWED_VIDEO_EXTS)}"
            ),
        )

    job_id = storage.new_job_id()
    ws = workspace_for(job_id)
    video_path = ws / f"input{suffix}"

    size = 0
    with video_path.open("wb") as out:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                shutil.rmtree(ws, ignore_errors=True)
                raise HTTPException(
                    status_code=413, detail="Video exceeds 500 MB limit"
                )
            out.write(chunk)
    if size == 0:
        shutil.rmtree(ws, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Empty video file")

    cfg = pipeline.PipelineConfig()
    if config:
        try:
            overrides = ModelConfig.model_validate(json.loads(config))
        except Exception as exc:
            shutil.rmtree(ws, ignore_errors=True)
            raise HTTPException(
                status_code=400, detail=f"Invalid config JSON: {exc}"
            ) from exc
        for k, v in overrides.model_dump(exclude_none=True).items():
            setattr(cfg, k, v)

    created_at = storage.now_iso()
    storage.save_analysis(
        DB_PATH,
        job_id,
        AnalysisStatus.processing.value,
        MODEL_VERSION,
        video.filename,
        cfg.__dict__,
        None,
        created_at,
    )

    try:
        result = pipeline.run_pipeline(str(video_path), cfg)
    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        storage.save_analysis(
            DB_PATH,
            job_id,
            AnalysisStatus.failed.value,
            MODEL_VERSION,
            video.filename,
            cfg.__dict__,
            {"error": str(exc)},
            created_at,
            storage.now_iso(),
        )
        shutil.rmtree(ws, ignore_errors=True)
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {exc}"
        ) from exc

    storage.save_analysis(
        DB_PATH,
        job_id,
        AnalysisStatus.completed.value,
        MODEL_VERSION,
        video.filename,
        cfg.__dict__,
        result,
        created_at,
        storage.now_iso(),
    )
    # Privacy: delete the raw video immediately after processing
    # (ARCHITECTURE.md section 8: avoid saving raw video by default).
    shutil.rmtree(ws, ignore_errors=True)

    row = storage.get_analysis(DB_PATH, job_id)
    assert row is not None
    return _result_from_row(row)


@app.get("/api/v1/analyses")
def list_analyses() -> list[dict]:
    return storage.list_analyses(DB_PATH)


@app.get("/api/v1/analyses/{job_id}", response_model=AnalysisResult)
def get_analysis(job_id: str) -> AnalysisResult:
    row = storage.get_analysis(DB_PATH, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _result_from_row(row)


@app.get("/api/v1/analyses/{job_id}/frames")
def get_frames(job_id: str) -> list[FrameResult]:
    row = storage.get_analysis(DB_PATH, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    result = json.loads(row["result_json"] or "{}")
    return [FrameResult(**f) for f in result.get("frames", [])]


@app.patch(
    "/api/v1/analyses/{job_id}/landmarks/{frame_id}",
    response_model=CorrectionOut,
)
def correct_landmarks(
    job_id: str, frame_id: str, body: CorrectionIn
) -> CorrectionOut:
    """Submit manually corrected landmark points for one frame.

    Recomputes that frame's measurements and the aggregate FI/category.
    Corrections are logged for audit (ARCHITECTURE.md section 8).
    """
    row = storage.get_analysis(DB_PATH, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    result = json.loads(row["result_json"] or "{}")
    frames = result.get("frames", [])
    target = next((f for f in frames if f["frame_id"] == frame_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Frame not found")

    old_lm = target.get("landmarks", {}) or {}
    new_lm = body.landmarks

    merged = LandmarksModel(
        nasion=new_lm.nasion or old_lm.get("nasion"),
        menton=new_lm.menton or old_lm.get("menton"),
        zygion_right=new_lm.zygion_right or old_lm.get("zygion_right"),
        zygion_left=new_lm.zygion_left or old_lm.get("zygion_left"),
    )

    for name in ("nasion", "menton", "zygion_right", "zygion_left"):
        old_pt = old_lm.get(name)
        new_pt = getattr(merged, name)
        if new_pt is not None:
            storage.log_correction(
                DB_PATH,
                job_id,
                frame_id,
                name,
                old_pt.get("x") if old_pt else None,
                old_pt.get("y") if old_pt else None,
                new_pt.x,
                new_pt.y,
            )

    lm = report_builder.schema_to_landmarks(merged)
    if lm is None:
        raise HTTPException(
            status_code=400, detail="All four landmarks are required"
        )

    measurements = report_builder.measurements_from_landmarks(lm)
    target["landmarks"] = merged.model_dump()
    target["measurements"] = measurements.model_dump()
    target["status"] = FrameQuality.accepted.value
    target["rejection"] = None

    updated_frames = [FrameResult(**f) for f in frames]
    aggregate = report_builder.build_aggregate(updated_frames)
    result["aggregate"] = aggregate.model_dump()
    result["representative_frame_id"] = report_builder.representative_frame_id(
        updated_frames
    )
    result["frames"] = [f.model_dump() for f in updated_frames]

    storage.save_analysis(
        DB_PATH,
        job_id,
        AnalysisStatus.completed.value,
        row["model_version"],
        row.get("filename"),
        json.loads(row["config_json"] or "{}"),
        result,
        row["created_at"],
        row.get("completed_at"),
    )

    updated = next(f for f in updated_frames if f.frame_id == frame_id)
    return CorrectionOut(job_id=job_id, frame=updated, aggregate=aggregate)


@app.delete("/api/v1/analyses/{job_id}", status_code=204)
def delete_analysis(job_id: str) -> None:
    row = storage.get_analysis(DB_PATH, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    storage.delete_analysis(DB_PATH, job_id)
    ws = WORKSPACE_DIR / job_id
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    return None

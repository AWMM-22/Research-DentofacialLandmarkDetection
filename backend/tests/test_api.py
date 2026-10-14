"""API integration tests using FastAPI TestClient.

Uses synthetic videos. A blank/synthetic clip will not contain a
detectable face, so the pipeline must reject frames with an explicit
`no_face` reason and the aggregate must remain None — never a fabricated
measurement.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import storage  # noqa: E402
from app.services.storage import init_db  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    # Point storage at a temp DB for isolation.
    import app.main as main_mod

    db = tmp_path_factory.mktemp("db") / "test.sqlite3"
    main_mod.DB_PATH = db
    init_db(db)
    with TestClient(app) as c:
        yield c


def _synthetic_video_bytes(frames: int = 10, size: tuple[int, int] = (320, 240)) -> bytes:
    """Render a short synthetic clip to an in-memory mp4."""
    w, h = size
    buf = io.BytesIO()
    # Use a temporary file because VideoWriter needs a real path.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for i in range(frames):
        img = np.full((h, w, 3), 128, dtype=np.uint8)
        cv2.circle(img, (w // 2, h // 2), 40 + i, (200, 180, 160), -1)
        writer.write(img)
    writer.release()
    data = Path(path).read_bytes()
    Path(path).unlink(missing_ok=True)
    return data


class TestHealth:
    def test_health(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "model_version" in body


class TestCreateAnalysisValidation:
    def test_rejects_wrong_extension(self, client: TestClient):
        r = client.post(
            "/api/v1/analyses",
            files={"video": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert r.status_code == 400

    def test_rejects_empty_file(self, client: TestClient):
        r = client.post(
            "/api/v1/analyses",
            files={"video": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
        )
        assert r.status_code == 400

    def test_rejects_bad_config(self, client: TestClient):
        video = _synthetic_video_bytes()
        r = client.post(
            "/api/v1/analyses",
            files={"video": ("clip.mp4", io.BytesIO(video), "video/mp4")},
            params={"config": '{"sample_fps": -1}'},
        )
        assert r.status_code == 400


class TestAnalysisLifecycle:
    def test_full_analysis_synthetic_video(self, client: TestClient):
        """A synthetic clip has no real face: pipeline must complete with
        explicit rejections and a None aggregate FI — never a guess."""
        video = _synthetic_video_bytes(frames=12)
        r = client.post(
            "/api/v1/analyses",
            files={"video": ("clip.mp4", io.BytesIO(video), "video/mp4")},
            params={"config": '{"sample_fps": 2, "max_frames": 6}'},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["status"] == "completed"
        assert body["aggregate"]["facial_index"] is None
        assert body["aggregate"]["accepted_frame_count"] == 0
        # Every frame must carry an explicit rejection reason.
        for f in body["frames"]:
            assert f["status"] in ("rejected", "flagged", "accepted")
            if f["status"] == "rejected":
                assert f["rejection"] is not None
                assert f["rejection"]["reason"]
        job_id = body["job_id"]

        # GET endpoints
        r2 = client.get(f"/api/v1/analyses/{job_id}")
        assert r2.status_code == 200
        assert r2.json()["job_id"] == job_id

        r3 = client.get(f"/api/v1/analyses/{job_id}/frames")
        assert r3.status_code == 200
        assert isinstance(r3.json(), list)

        # DELETE
        r4 = client.delete(f"/api/v1/analyses/{job_id}")
        assert r4.status_code == 204
        r5 = client.get(f"/api/v1/analyses/{job_id}")
        assert r5.status_code == 404

    def test_missing_job_404(self, client: TestClient):
        r = client.get("/api/v1/analyses/does-not-exist")
        assert r.status_code == 404

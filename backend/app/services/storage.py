"""SQLite persistence for analysis metadata and results.

Implements ARCHITECTURE.md section 8. Raw video is NOT retained by default;
only job metadata, per-frame results, landmarks, and correction logs are
stored. Video files live in a workspace directory for the duration of
processing and can be deleted with the analysis.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import WORKSPACE_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    model_version TEXT NOT NULL,
    filename TEXT,
    config_json TEXT,
    result_json TEXT
);
CREATE TABLE IF NOT EXISTS frame_result (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES analysis(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    quality_status TEXT NOT NULL,
    height REAL,
    width REAL,
    facial_index REAL,
    rejection_reason TEXT
);
CREATE TABLE IF NOT EXISTS landmark (
    id TEXT PRIMARY KEY,
    frame_result_id TEXT NOT NULL REFERENCES frame_result(id) ON DELETE CASCADE,
    point_name TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    confidence REAL,
    source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS correction_log (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES analysis(id) ON DELETE CASCADE,
    frame_id TEXT NOT NULL,
    point_name TEXT NOT NULL,
    old_x REAL,
    old_y REAL,
    new_x REAL NOT NULL,
    new_y REAL NOT NULL,
    corrected_at TEXT NOT NULL
);
"""

_lock = threading.Lock()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    with _lock:
        conn = _connect(db_path)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()


def new_job_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def workspace_for(job_id: str) -> Path:
    d = WORKSPACE_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_analysis(
    db_path: Path,
    job_id: str,
    status: str,
    model_version: str,
    filename: Optional[str],
    config: dict,
    result: Optional[dict],
    created_at: str,
    completed_at: Optional[str] = None,
) -> None:
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO analysis
                    (id, created_at, completed_at, status, model_version, filename, config_json, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    completed_at=excluded.completed_at,
                    status=excluded.status,
                    result_json=excluded.result_json
                """,
                (
                    job_id,
                    created_at,
                    completed_at,
                    status,
                    model_version,
                    filename,
                    json.dumps(config),
                    json.dumps(result) if result is not None else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def get_analysis(db_path: Path, job_id: str) -> Optional[dict]:
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute(
                "SELECT * FROM analysis WHERE id = ?", (job_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def list_analyses(db_path: Path) -> list[dict]:
    with _lock:
        conn = _connect(db_path)
        try:
            rows = conn.execute(
                "SELECT id, created_at, status, filename FROM analysis ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def delete_analysis(db_path: Path, job_id: str) -> bool:
    with _lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute("DELETE FROM analysis WHERE id = ?", (job_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def log_correction(
    db_path: Path,
    job_id: str,
    frame_id: str,
    point_name: str,
    old_x: Optional[float],
    old_y: Optional[float],
    new_x: float,
    new_y: float,
) -> None:
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO correction_log
                    (id, analysis_id, frame_id, point_name, old_x, old_y, new_x, new_y, corrected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    job_id,
                    frame_id,
                    point_name,
                    old_x,
                    old_y,
                    new_x,
                    new_y,
                    now_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

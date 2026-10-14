import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  LANDMARK_COLORS,
  LANDMARK_NAMES,
  LANDMARK_SHORT,
  type AnalysisResult,
  type FrameResult,
  type LandmarkName,
  type LandmarksModel,
} from "../api/types";
import { correctLandmarks } from "../api/client";

interface Props {
  initial: AnalysisResult;
  onBack: () => void;
}

const PAD = 40; // label padding around each landmark

export default function ResultPage({ initial, onBack }: Props) {
  const [result, setResult] = useState<AnalysisResult>(initial);
  const [selectedFrameId, setSelectedFrameId] = useState<string | null>(
    initial.representative_frame_id ?? initial.frames[0]?.frame_id ?? null,
  );
  const [draft, setDraft] = useState<LandmarksModel | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const frames = result.frames;
  const selected =
    frames.find((f) => f.frame_id === selectedFrameId) ?? frames[0] ?? null;

  // Reset the editable draft whenever a different frame is selected.
  useEffect(() => {
    if (!selected) {
      setDraft(null);
      return;
    }
    setDraft({ ...selected.landmarks });
    setSaveError(null);
  }, [selected?.frame_id]); // eslint-disable-line react-hooks/exhaustive-deps

  const movePoint = useCallback(
    (name: LandmarkName, x: number, y: number) => {
      setDraft((prev) => {
        if (!prev) return prev;
        const existing = prev[name];
        return {
          ...prev,
          [name]: {
            x,
            y,
            confidence: existing?.confidence ?? null,
            source: "manual" as const,
          },
        };
      });
    },
    [],
  );

  async function saveCorrections() {
    if (!selected || !draft) return;
    setSaving(true);
    setSaveError(null);
    try {
      const out = await correctLandmarks(result.job_id, selected.frame_id, draft);
      setResult((prev) => ({
        ...prev,
        frames: prev.frames.map((f) =>
          f.frame_id === out.frame.frame_id ? out.frame : f,
        ),
        aggregate: out.aggregate,
      }));
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  const accepted = frames.filter((f) => f.status === "accepted");
  const fi = result.aggregate.facial_index;
  const outside = result.aggregate.category.startsWith("Outside");

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>2. Review & correct</h2>
        <button onClick={onBack}>← Analyse another video</button>
      </div>

      {/* ---- Aggregate report ---- */}
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Facial Index report</h3>
        <div className="row" style={{ alignItems: "baseline", gap: "1.5rem" }}>
          <div>
            <div className="muted">Aggregate FI (median)</div>
            <div className="fi-big">{fi != null ? fi.toFixed(1) : "—"}</div>
          </div>
          <div>
            <div className="muted">Category</div>
            <span className={`category-pill${outside ? " outside" : ""}`}>
              {result.aggregate.category}
            </span>
          </div>
          <div>
            <div className="muted">Accepted frames</div>
            <div>
              {result.aggregate.accepted_frame_count} / {result.quality.sampled}
            </div>
          </div>
          {result.aggregate.median_height != null && (
            <div>
              <div className="muted">H / W (px, medians)</div>
              <div>
                {result.aggregate.median_height.toFixed(1)} /{" "}
                {result.aggregate.median_width?.toFixed(1)}
              </div>
            </div>
          )}
        </div>

        <table style={{ marginTop: "1rem" }}>
          <tbody>
            <tr>
              <th>Model</th>
              <td>
                {result.model.name} <code>{result.model.version}</code>
              </td>
            </tr>
            <tr>
              <th>Frames</th>
              <td>
                accepted {result.quality.accepted} · flagged{" "}
                {result.quality.flagged} · rejected {result.quality.rejected} ·
                sampled {result.quality.sampled}
              </td>
            </tr>
            {Object.keys(result.quality.rejection_reasons).length > 0 && (
              <tr>
                <th>Rejection reasons</th>
                <td>
                  {Object.entries(result.quality.rejection_reasons)
                    .map(([k, v]) => `${k} ×${v}`)
                    .join(" · ")}
                </td>
              </tr>
            )}
          </tbody>
        </table>

        {result.warnings.length > 0 && (
          <div className="warnings">
            <strong>Warnings</strong>
            <ul>
              {result.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* ---- Frame picker ---- */}
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Accepted / flagged frames</h3>
        {accepted.length === 0 && result.frames.filter((f) => f.status === "flagged").length === 0 ? (
          <p className="muted">
            No frames passed the quality gate. Upload a clearer, near-frontal
            video with a visible face.
          </p>
        ) : (
          <div className="row">
            {[...accepted, ...frames.filter((f) => f.status === "flagged")].map(
              (f) => (
                <button
                  key={f.frame_id}
                  onClick={() => setSelectedFrameId(f.frame_id)}
                  style={{
                    outline:
                      f.frame_id === selectedFrameId
                        ? "2px solid #2563eb"
                        : undefined,
                  }}
                >
                  #{f.frame_index} {(f.measurements?.facial_index ?? 0).toFixed(1)}
                  <span className={`badge ${f.status}`} style={{ marginLeft: 6 }}>
                    {f.status}
                  </span>
                </button>
              ),
            )}
          </div>
        )}
      </div>

      {/* ---- Landmark editor ---- */}
      {selected && draft && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>
            Landmarks — frame #{selected.frame_index} (
            {selected.timestamp_ms} ms)
          </h3>
          <p className="muted">
            Drag any point to correct it. Menton ≠ pogonion/gnathion; zygion is
            an anthropometric point, not just the outer cheek outline — see
            ARCHITECTURE.md §4 before trusting these.
          </p>
          <LandmarkCanvas
            frame={selected}
            draft={draft}
            onMove={movePoint}
          />
          <div className="legend">
            {LANDMARK_NAMES.map((n) => (
              <span key={n}>
                <span
                  className="dot"
                  style={{ background: LANDMARK_COLORS[n] }}
                />
                {LANDMARK_SHORT[n]}{" "}
                {draft[n]?.source === "manual" ? "(manual)" : ""}
              </span>
            ))}
          </div>
          {selected.measurements && (
            <p className="muted" style={{ marginTop: "0.5rem" }}>
              H = {selected.measurements.height.toFixed(1)} px · W ={" "}
              {selected.measurements.width.toFixed(1)} px · FI ={" "}
              {selected.measurements.facial_index.toFixed(2)} (unrounded; round
              only for display)
            </p>
          )}
          <div className="row" style={{ marginTop: "0.75rem" }}>
            <button
              className="primary"
              onClick={saveCorrections}
              disabled={saving}
            >
              {saving ? "Saving…" : "Save manual corrections"}
            </button>
            {saveError && <span className="error">{saveError}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Canvas overlay + drag editor
// ---------------------------------------------------------------------------
interface CanvasProps {
  frame: FrameResult;
  draft: LandmarksModel;
  onMove: (name: LandmarkName, x: number, y: number) => void;
}

function LandmarkCanvas({ frame, draft, onMove }: CanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<LandmarkName | null>(null);

  // Determine the native frame size from landmarks/face_box (fallback 640x480).
  const native = useMemo(() => {
    let maxX = 640;
    let maxY = 480;
    for (const n of LANDMARK_NAMES) {
      const p = draft[n];
      if (p) {
        maxX = Math.max(maxX, p.x + PAD);
        maxY = Math.max(maxY, p.y + PAD);
      }
    }
    if (frame.face_box) {
      maxX = Math.max(maxX, frame.face_box[0] + frame.face_box[2] + PAD);
      maxY = Math.max(maxY, frame.face_box[1] + frame.face_box[3] + PAD);
    }
    return { w: Math.ceil(maxX), h: Math.ceil(maxY) };
  }, [draft, frame.face_box]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { w, h } = native;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 0, w, h);

    // Face box outline.
    if (frame.face_box) {
      const [bx, by, bw, bh] = frame.face_box;
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(bx, by, bw, bh);
      ctx.setLineDash([]);
    }

    // Connections: nasion-menton (height), zygion_right-zygion_left (width).
    const n = draft.nasion;
    const m = draft.menton;
    const zr = draft.zygion_right;
    const zl = draft.zygion_left;
    ctx.lineWidth = 2;
    if (n && m) {
      ctx.strokeStyle = "#fbbf24";
      ctx.beginPath();
      ctx.moveTo(n.x, n.y);
      ctx.lineTo(m.x, m.y);
      ctx.stroke();
    }
    if (zr && zl) {
      ctx.strokeStyle = "#38bdf8";
      ctx.beginPath();
      ctx.moveTo(zr.x, zr.y);
      ctx.lineTo(zl.x, zl.y);
      ctx.stroke();
    }

    // Points + labels.
    ctx.font = "13px sans-serif";
    for (const name of LANDMARK_NAMES) {
      const p = draft[name];
      if (!p) continue;
      const color = LANDMARK_COLORS[name];
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#fff";
      ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.fillText(LANDMARK_SHORT[name], p.x + 10, p.y - 10);
    }
  }, [draft, frame.face_box, native]);

  useEffect(() => {
    draw();
  }, [draw]);

  function toNative(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * native.w;
    const y = ((e.clientY - rect.top) / rect.height) * native.h;
    return { x, y };
  }

  function pickName(x: number, y: number): LandmarkName | null {
    let best: LandmarkName | null = null;
    let bestD = 15; // grab radius in native px
    for (const name of LANDMARK_NAMES) {
      const p = draft[name];
      if (!p) continue;
      const d = Math.hypot(p.x - x, p.y - y);
      if (d < bestD) {
        bestD = d;
        best = name;
      }
    }
    return best;
  }

  return (
    <canvas
      ref={canvasRef}
      className="landmark-canvas"
      onPointerDown={(e) => {
        const { x, y } = toNative(e);
        const name = pickName(x, y);
        if (name) {
          dragRef.current = name;
          (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
        }
      }}
      onPointerMove={(e) => {
        if (!dragRef.current) return;
        const { x, y } = toNative(e);
        onMove(dragRef.current, x, y);
      }}
      onPointerUp={() => {
        dragRef.current = null;
      }}
    />
  );
}

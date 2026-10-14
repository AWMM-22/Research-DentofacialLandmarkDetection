import { useRef, useState } from "react";
import { createAnalysis } from "../api/client";
import type { AnalysisResult } from "../api/types";

interface Props {
  onCompleted: (result: AnalysisResult) => void;
}

export default function UploadPage({ onCompleted }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const result = await createAnalysis(file, { sample_fps: 2, max_frames: 60 });
      onCompleted(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>1. Upload a near-frontal facial video</h2>
      <p className="muted">
        Accepts mp4 / mov / avi / mkv / webm, up to 500&nbsp;MB. Frames are
        quality-gated; unsuitable frames are rejected rather than measured.
      </p>
      <div className="row">
        <input
          ref={fileRef}
          type="file"
          accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
          onChange={(e) => {
            setFile(e.target.files?.[0] ?? null);
            setError(null);
          }}
        />
        <button className="primary" onClick={submit} disabled={!file || busy}>
          {busy ? "Analysing…" : "Analyse video"}
        </button>
      </div>
      {busy && (
        <p className="muted" style={{ marginTop: "0.75rem" }}>
          Decoding, quality-gating, estimating landmarks, and measuring. This
          runs synchronously on the server and may take a while for long clips.
        </p>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
}

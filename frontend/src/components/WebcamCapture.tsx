import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  onRecording: (file: File) => void;
}

type CamState = "idle" | "starting" | "live" | "recording" | "error";

export default function WebcamCapture({ onRecording }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);

  const [state, setState] = useState<CamState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const stopStream = useCallback(() => {
    if (timerRef.current != null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      try {
        recorderRef.current.stop();
      } catch {
        /* already stopped */
      }
    }
    recorderRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setElapsed(0);
  }, []);

  useEffect(() => stopStream, [stopStream]);

  async function start() {
    setState("starting");
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setState("live");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not access webcam");
      setState("error");
    }
  }

  function startRecording() {
    const stream = streamRef.current;
    if (!stream) return;
    chunksRef.current = [];
    const mime = pickMime();
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    rec.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    rec.onstop = () => {
      const type = rec.mimeType || "video/webm";
      const blob = new Blob(chunksRef.current, { type });
      const ext = type.includes("mp4") ? "mp4" : "webm";
      const file = new File([blob], `webcam-${Date.now()}.${ext}`, { type });
      stopStream();
      setState("idle");
      onRecording(file);
    };
    recorderRef.current = rec;
    rec.start(250); // collect chunks every 250ms
    setState("recording");
    setElapsed(0);
    timerRef.current = window.setInterval(() => setElapsed((s) => s + 1), 1000);
  }

  function stopRecording() {
    recorderRef.current?.stop();
  }

  return (
    <div>
      <div className="row" style={{ marginBottom: "0.5rem" }}>
        {state === "idle" && (
          <button className="primary" onClick={start}>
            Start webcam
          </button>
        )}
        {state === "starting" && <span className="muted">Requesting camera…</span>}
        {state === "live" && (
          <>
            <button className="primary" onClick={startRecording}>
              ● Record & analyse
            </button>
            <button onClick={stopStream}>Cancel</button>
          </>
        )}
        {state === "recording" && (
          <>
            <button className="primary" onClick={stopRecording}>
              ■ Stop & analyse ({elapsed}s)
            </button>
            <span className="muted">Recording… face the camera near-frontally.</span>
          </>
        )}
        {error && <span className="error">{error}</span>}
      </div>
      <video
        ref={videoRef}
        playsInline
        muted
        style={{
          width: "100%",
          maxWidth: 640,
          borderRadius: 8,
          border: "1px solid var(--border)",
          background: "#000",
          display: state === "idle" && !streamRef.current ? "none" : "block",
        }}
      />
    </div>
  );
}

function pickMime(): string | undefined {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  for (const c of candidates) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(c)) {
      return c;
    }
  }
  return undefined;
}

import type {
  AnalysisResult,
  CorrectionOut,
  HealthResponse,
  LandmarksModel,
} from "./types";

const BASE = "";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return handle(await fetch(`${BASE}/health`));
}

export async function createAnalysis(
  file: File,
  config?: Record<string, unknown>,
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("video", file, file.name);
  let url = `${BASE}/api/v1/analyses`;
  if (config && Object.keys(config).length > 0) {
    url += `?config=${encodeURIComponent(JSON.stringify(config))}`;
  }
  return handle(await fetch(url, { method: "POST", body: form }));
}

export async function getAnalysis(jobId: string): Promise<AnalysisResult> {
  return handle(await fetch(`${BASE}/api/v1/analyses/${jobId}`));
}

export async function correctLandmarks(
  jobId: string,
  frameId: string,
  landmarks: LandmarksModel,
): Promise<CorrectionOut> {
  return handle(
    await fetch(`${BASE}/api/v1/analyses/${jobId}/landmarks/${frameId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame_id: frameId, landmarks }),
    }),
  );
}

export async function deleteAnalysis(jobId: string): Promise<void> {
  await handle(await fetch(`${BASE}/api/v1/analyses/${jobId}`, { method: "DELETE" }));
}

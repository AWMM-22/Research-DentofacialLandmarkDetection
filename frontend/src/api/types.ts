// Types mirroring backend/app/schemas.py. Keep in sync.

export type AnalysisStatus = "pending" | "processing" | "completed" | "failed";
export type FrameQuality = "accepted" | "rejected" | "flagged";
export type LandmarkSource = "model" | "manual";

export interface Point {
  x: number;
  y: number;
}

export interface LandmarkPoint extends Point {
  confidence?: number | null;
  source: LandmarkSource;
}

export interface LandmarksModel {
  nasion?: LandmarkPoint | null;
  menton?: LandmarkPoint | null;
  zygion_right?: LandmarkPoint | null;
  zygion_left?: LandmarkPoint | null;
}

export type LandmarkName = keyof LandmarksModel;

export const LANDMARK_NAMES: LandmarkName[] = [
  "nasion",
  "menton",
  "zygion_right",
  "zygion_left",
];

export const LANDMARK_LABELS: Record<LandmarkName, string> = {
  nasion: "N′ (soft-tissue nasion)",
  menton: "Me′ (soft-tissue menton)",
  zygion_right: "Zyᵣ (right zygion)",
  zygion_left: "Zyₗ (left zygion)",
};

export const LANDMARK_SHORT: Record<LandmarkName, string> = {
  nasion: "N′",
  menton: "Me′",
  zygion_right: "Zyᵣ",
  zygion_left: "Zyₗ",
};

export const LANDMARK_COLORS: Record<LandmarkName, string> = {
  nasion: "#e74c3c",
  menton: "#e67e22",
  zygion_right: "#3498db",
  zygion_left: "#9b59b6",
};

export interface FrameMeasurements {
  height: number;
  width: number;
  facial_index: number;
}

export interface FrameRejection {
  reason: string;
  detail?: string | null;
}

export interface FrameResult {
  frame_id: string;
  frame_index: number;
  timestamp_ms: number;
  status: FrameQuality;
  landmarks: LandmarksModel;
  measurements?: FrameMeasurements | null;
  rejection?: FrameRejection | null;
  blur_variance?: number | null;
  yaw_deg?: number | null;
  pitch_deg?: number | null;
  roll_deg?: number | null;
  face_box?: [number, number, number, number] | null;
}

export interface QualitySummary {
  sampled: number;
  accepted: number;
  rejected: number;
  flagged: number;
  rejection_reasons: Record<string, number>;
}

export interface AggregateResult {
  facial_index: number | null;
  category: string;
  median_height: number | null;
  median_width: number | null;
  accepted_frame_count: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  note: string;
}

export interface AnalysisResult {
  job_id: string;
  status: AnalysisStatus;
  created_at: string;
  completed_at?: string | null;
  filename?: string | null;
  model: ModelInfo;
  quality: QualitySummary;
  aggregate: AggregateResult;
  representative_frame_id?: string | null;
  frames: FrameResult[];
  warnings: string[];
}

export interface CorrectionOut {
  job_id: string;
  frame: FrameResult;
  aggregate: AggregateResult;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version?: string | null;
}

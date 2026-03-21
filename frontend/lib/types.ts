export interface Run {
  id: string;
  intent: string;
  status: "running" | "completed" | "failed";
  total_images?: number;
  usable_count?: number;
  recoverable_count?: number;
  unusable_count?: number;
  error_count?: number;
  overall_score?: number;
  created_at?: string;
}

export interface ImageResult {
  image_id: string;
  image_path: string;
  verdict: "usable" | "recoverable" | "unusable" | "error";
  scores: Record<string, number>;
  errors: string[];
  flags: string[];
}

export interface PipelineEvent {
  type: string;
  module: string;
  timestamp: string;
  [key: string]: unknown;
}

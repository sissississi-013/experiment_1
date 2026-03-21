const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers: { "Content-Type": "application/json", ...options?.headers } });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function listRuns(limit = 20) { return fetchAPI<any[]>(`/api/runs?limit=${limit}`); }
export async function getRun(runId: string) { return fetchAPI<any>(`/api/runs/${runId}`); }
export async function createRun(body: { intent: string; dataset_description?: string }) {
  return fetchAPI<{ run_id: string; status: string }>("/api/runs", { method: "POST", body: JSON.stringify(body) });
}
export async function getRunImages(runId: string, verdict?: string) {
  const params = verdict ? `?verdict=${verdict}` : "";
  return fetchAPI<any[]>(`/api/runs/${runId}/images${params}`);
}
export function getExportUrl(runId: string, filter = "usable") {
  return `${API_BASE}/api/runs/${runId}/export?filter=${filter}`;
}

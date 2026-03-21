import type { PipelineEvent } from "./types";

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export function connectToRun(runId: string, onEvent: (event: PipelineEvent) => void, onClose?: () => void): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/api/runs/${runId}/stream`);
  ws.onmessage = (msg) => { try { const e = JSON.parse(msg.data); if (e.type !== "ping") onEvent(e); } catch {} };
  ws.onclose = () => onClose?.();
  ws.onerror = () => onClose?.();
  return ws;
}

"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { listRuns } from "@/lib/api";

interface Run {
  id: string;
  intent: string;
  status: string;
  total_images?: number;
  usable_count?: number;
  overall_score?: number;
  created_at?: string;
}

export default function HistoryPage() {
  const router = useRouter();
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadRuns();
  }, []);

  async function loadRuns() {
    try {
      const data = await listRuns(50);
      setRuns(data);
    } catch (err) {
      setError("Failed to load run history. Is the API server running?");
    } finally {
      setLoading(false);
    }
  }

  function formatDate(dateStr?: string) {
    if (!dateStr) return "—";
    try {
      return new Date(dateStr).toLocaleString("en-US", {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
      });
    } catch { return dateStr; }
  }

  return (
    <div style={{ padding: "32px" }}>
      <div style={{ marginBottom: "32px" }}>
        <h2 style={{ fontSize: "28px", fontWeight: 700, letterSpacing: "-0.02em", marginBottom: "4px" }}>
          Run History
        </h2>
        <p style={{ color: "#6E6E73", fontSize: "15px" }}>
          All pipeline runs stored in Neon Postgres
        </p>
      </div>

      {loading && <p style={{ color: "#6E6E73" }}>Loading...</p>}
      {error && <p style={{ color: "#FF3B30", fontSize: "14px" }}>{error}</p>}

      {!loading && runs.length === 0 && !error && (
        <p style={{ color: "#6E6E73" }}>No runs yet. Start one from the New Run page.</p>
      )}

      {runs.length > 0 && (
        <div>
          {/* Header */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "80px 1fr 90px 70px 70px 130px",
            gap: "12px",
            padding: "8px 12px",
            fontSize: "11px",
            fontWeight: 500,
            color: "#6E6E73",
            borderBottom: "1px solid var(--border)",
            marginBottom: "4px",
          }}>
            <div>Run ID</div>
            <div>Intent</div>
            <div style={{ textAlign: "center" }}>Status</div>
            <div style={{ textAlign: "right" }}>Images</div>
            <div style={{ textAlign: "right" }}>Score</div>
            <div style={{ textAlign: "right" }}>Time</div>
          </div>

          {/* Rows */}
          {runs.map((run) => (
            <div
              key={run.id}
              onClick={() => router.push(`/live/${run.id}`)}
              style={{
                display: "grid",
                gridTemplateColumns: "80px 1fr 90px 70px 70px 130px",
                gap: "12px",
                padding: "10px 12px",
                fontSize: "13px",
                borderRadius: "8px",
                cursor: "pointer",
                transition: "background 0.2s",
                marginBottom: "2px",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "var(--surface)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <div style={{ color: "#0071E3", fontFamily: "SF Mono, Menlo, monospace", fontSize: "12px" }}>
                {run.id}
              </div>
              <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {run.intent}
              </div>
              <div style={{
                textAlign: "center",
                fontSize: "11px",
                fontWeight: 500,
                color: run.status === "completed" ? "#34C759" : run.status === "failed" ? "#FF3B30" : "#FF9F0A",
              }}>
                {run.status}
              </div>
              <div style={{ textAlign: "right", color: "#6E6E73" }}>
                {run.total_images ?? "—"}
              </div>
              <div style={{ textAlign: "right", fontWeight: 600 }}>
                {run.overall_score != null ? run.overall_score.toFixed(2) : "—"}
              </div>
              <div style={{ textAlign: "right", color: "#6E6E73", fontSize: "11px" }}>
                {formatDate(run.created_at)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

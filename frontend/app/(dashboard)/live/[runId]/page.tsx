"use client";
import { useEffect, useState, useRef, useCallback } from "react";
import { useParams } from "next/navigation";
import { connectToRun } from "@/lib/ws";
import { getRun, getRunImages, getImageUrl } from "@/lib/api";
import type { PipelineEvent } from "@/lib/types";
import StatsBar from "@/components/StatsBar";
import ProgressLog from "@/components/ProgressLog";
import ImageGrid from "@/components/ImageGrid";
import ExportDropdown from "@/components/ExportDropdown";

interface Img {
  image_id: string;
  image_path?: string;
  verdict: string;
  scores: Record<string, number>;
  errors?: string[];
}

const verdictColors: Record<string, string> = {
  usable: "#34C759",
  recoverable: "#FF9F0A",
  unusable: "#FF3B30",
  error: "#FF453A",
};

const verdictLabels: Record<string, string> = {
  usable: "All quality and content checks passed",
  recoverable: "Failed one check — may be fixable",
  unusable: "Failed multiple checks",
  error: "Tool errors prevented evaluation",
};

export default function LiveViewPage() {
  const params = useParams();
  const runId = params.runId as string;
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [images, setImages] = useState<Img[]>([]);
  const [stats, setStats] = useState({ processed: 0, total: 0, usable: 0, recoverable: 0, unusable: 0, elapsed: 0 });
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState("");
  const [selectedImg, setSelectedImg] = useState<Img | null>(null);
  const [intent, setIntent] = useState("");
  const startTime = useRef(Date.now());
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const completedRef = useRef(false);

  useEffect(() => {
    const ws = connectToRun(runId, (event) => {
      setEvents((prev) => [...prev, event]);
      handleEvent(event);
    }, () => loadCompletedRun());

    timerRef.current = setInterval(() => {
      if (!completedRef.current) {
        setStats((prev) => ({ ...prev, elapsed: (Date.now() - startTime.current) / 1000 }));
      }
    }, 100);

    return () => {
      ws.close();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [runId]);

  const markCompleted = useCallback((finalElapsed?: number) => {
    completedRef.current = true;
    setCompleted(true);
    if (timerRef.current) clearInterval(timerRef.current);
    if (finalElapsed != null) {
      setStats((p) => ({ ...p, elapsed: finalElapsed }));
    }
  }, []);

  function handleEvent(event: PipelineEvent) {
    if (event.type === "ImageProgress") {
      const e = event as any;
      setStats((p) => ({ ...p, processed: e.current, total: e.total }));
    }
    if (event.type === "ImageVerdict") {
      const e = event as any;
      setImages((p) => [...p, {
        image_id: e.image_id,
        image_path: e.image_path,
        verdict: e.verdict,
        scores: e.scores || {},
        errors: e.errors || [],
      }]);
      setStats((p) => ({
        ...p,
        usable: p.usable + (e.verdict === "usable" ? 1 : 0),
        recoverable: p.recoverable + (e.verdict === "recoverable" ? 1 : 0),
        unusable: p.unusable + (e.verdict === "unusable" ? 1 : 0),
      }));
    }
    if (event.type === "SpecGenerated") {
      setIntent((event as any).spec_summary || "");
    }
    if (event.type === "RunCompleted") {
      markCompleted();
    }
    if (event.type === "ModuleCompleted" && (event as any).module === "reporter") {
      markCompleted((event as any).duration_seconds);
    }
    if (event.type === "ModuleCompleted" && (event as any).module === "executor") {
      setStats((p) => ({ ...p, elapsed: (event as any).duration_seconds }));
    }
  }

  async function loadCompletedRun() {
    try {
      const run = await getRun(runId);
      if (run.status === "completed" || run.status === "failed") {
        markCompleted(0);
        setIntent(run.intent || "");
        setStats({
          processed: run.total_images || 0,
          total: run.total_images || 0,
          usable: run.usable_count || 0,
          recoverable: run.recoverable_count || 0,
          unusable: run.unusable_count || 0,
          elapsed: 0,
        });
        const imgs = await getRunImages(runId);
        setImages(imgs.map((img: any) => ({
          image_id: img.image_id,
          image_path: img.image_path,
          verdict: img.verdict,
          scores: typeof img.scores === "string" ? JSON.parse(img.scores) : img.scores || {},
          errors: typeof img.errors === "string" ? JSON.parse(img.errors) : img.errors || [],
        })));
      }
    } catch {
      setError("Run not found");
    }
  }

  const progress = stats.total > 0 ? (stats.processed / stats.total) * 100 : 0;

  function getFailedDimensions(img: Img): string[] {
    // Scores below typical thresholds
    const failed: string[] = [];
    for (const [dim, score] of Object.entries(img.scores)) {
      if (dim === "blur" && score < 0.4) failed.push("blur");
      else if (dim === "exposure" && score < 0.45) failed.push("exposure");
      else if (dim === "content" && score < 0.6) failed.push("content");
      else if (score < 0.5) failed.push(dim);
    }
    return failed;
  }

  function getVerdictExplanation(img: Img): string {
    if (img.verdict === "usable") return "This image passed all quality and content checks.";
    if (img.verdict === "error") return "Tool errors prevented proper evaluation. " + (img.errors?.join("; ") || "");
    const failed = getFailedDimensions(img);
    if (failed.length === 0) return verdictLabels[img.verdict] || "";
    const reasons = failed.map((d) => {
      const score = img.scores[d];
      if (d === "blur") return `Blur score ${score?.toFixed(2)} — image may be too blurry or soft`;
      if (d === "exposure") return `Exposure score ${score?.toFixed(2)} — image may be too dark or overexposed`;
      if (d === "content") return `Content score ${score?.toFixed(2)} — target object not confidently detected`;
      return `${d} score ${score?.toFixed(2)} — below threshold`;
    });
    return reasons.join(". ") + ".";
  }

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* Main content */}
      <div style={{ flex: 1, overflowY: "auto", padding: "24px" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px" }}>
          <div>
            <h2 style={{ fontSize: "22px", fontWeight: 700, letterSpacing: "-0.02em" }}>
              Run {runId}
              {!completed && (
                <span style={{ marginLeft: "8px", fontSize: "13px", fontWeight: 400, color: "#0071E3" }}>● Live</span>
              )}
              {completed && (
                <span style={{ marginLeft: "8px", fontSize: "13px", fontWeight: 400, color: "#34C759" }}>✓ Complete</span>
              )}
            </h2>
            {intent && <p style={{ color: "#6E6E73", fontSize: "13px", marginTop: "2px" }}>{intent}</p>}
          </div>
          {completed && <ExportDropdown runId={runId} />}
        </div>

        {/* Completion note */}
        {completed && stats.total > 0 && (
          <div style={{
            background: "var(--surface)",
            borderRadius: "12px",
            padding: "14px 18px",
            marginBottom: "16px",
            fontSize: "13px",
            color: "var(--text-secondary)",
          }}>
            <strong style={{ color: "var(--text-primary)" }}>{stats.usable} of {stats.total} images</strong> met all validation criteria.
            {stats.usable < stats.total && (
              <span> {stats.recoverable} recoverable (failed 1 check), {stats.unusable} unusable. Click an image below for details.</span>
            )}
          </div>
        )}

        <StatsBar {...stats} />

        {/* Progress bar */}
        <div style={{ height: "3px", background: "var(--surface)", borderRadius: "2px", marginBottom: "16px", overflow: "hidden" }}>
          <div style={{
            height: "100%", borderRadius: "2px",
            transition: "width 0.5s",
            width: `${progress}%`,
            background: completed ? "#34C759" : "linear-gradient(90deg, #0071E3, #00C7FF)",
          }} />
        </div>

        {error && <div style={{ color: "#FF3B30", fontSize: "13px", marginBottom: "16px" }}>{error}</div>}

        <ProgressLog events={events} />

        {/* Image grid — clickable */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "2px" }}>
          {images.map((img, i) => (
            <div
              key={`${img.image_id}-${i}`}
              onClick={() => setSelectedImg(img)}
              style={{
                aspectRatio: "1",
                background: "var(--surface)",
                borderRadius: "4px",
                position: "relative",
                overflow: "hidden",
                cursor: "pointer",
                border: selectedImg?.image_id === img.image_id ? "2px solid #0071E3" : "2px solid transparent",
                transition: "border-color 0.2s",
              }}
            >
              {img.image_path && (
                <img
                  src={getImageUrl(img.image_path)}
                  alt={img.image_id}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  loading="lazy"
                />
              )}
              <div style={{
                position: "absolute", top: "6px", right: "6px",
                width: "10px", height: "10px", borderRadius: "50%",
                background: verdictColors[img.verdict] || "#6E6E73",
                border: "1.5px solid rgba(0,0,0,0.3)",
              }} />
              <div style={{
                position: "absolute", bottom: 0, left: 0, right: 0,
                background: "linear-gradient(transparent, rgba(0,0,0,0.7))",
                padding: "16px 6px 4px",
                fontSize: "10px",
                color: "white",
                display: "flex",
                justifyContent: "space-between",
              }}>
                <span>{img.verdict}</span>
              </div>
            </div>
          ))}
          {/* Empty slots for in-progress */}
          {Array.from({ length: Math.max(0, (stats.total || 0) - images.length) }).map((_, i) => (
            <div key={`empty-${i}`} style={{ aspectRatio: "1", background: "var(--surface)", borderRadius: "4px", opacity: 0.2 }} />
          ))}
        </div>
      </div>

      {/* Detail panel */}
      {selectedImg && (
        <div style={{
          width: "340px",
          borderLeft: "1px solid var(--border)",
          padding: "24px",
          overflowY: "auto",
          flexShrink: 0,
        }}>
          {/* Image preview */}
          <div style={{
            width: "100%", aspectRatio: "4/3",
            background: "var(--surface)", borderRadius: "12px",
            overflow: "hidden", marginBottom: "16px",
          }}>
            {selectedImg.image_path && (
              <img
                src={getImageUrl(selectedImg.image_path)}
                alt={selectedImg.image_id}
                style={{ width: "100%", height: "100%", objectFit: "cover" }}
              />
            )}
          </div>

          {/* Filename + verdict */}
          <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "4px" }}>
            {selectedImg.image_id}.jpg
          </h3>
          <div style={{
            fontSize: "14px", fontWeight: 600,
            color: verdictColors[selectedImg.verdict] || "#6E6E73",
            marginBottom: "8px",
          }}>
            {selectedImg.verdict.charAt(0).toUpperCase() + selectedImg.verdict.slice(1)}
          </div>

          {/* Why this verdict */}
          <div style={{
            fontSize: "12px",
            color: "var(--text-secondary)",
            lineHeight: "1.5",
            marginBottom: "20px",
            background: "var(--surface)",
            borderRadius: "8px",
            padding: "10px 12px",
          }}>
            {getVerdictExplanation(selectedImg)}
          </div>

          {/* Score bars */}
          <div style={{ fontSize: "11px", fontWeight: 600, color: "#6E6E73", marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Quality Scores
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "20px" }}>
            {Object.entries(selectedImg.scores).map(([dim, score]) => {
              const passed = dim === "blur" ? score >= 0.4 : dim === "exposure" ? score >= 0.45 : dim === "content" ? score >= 0.6 : score >= 0.5;
              return (
                <div key={dim}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "3px" }}>
                    <span style={{ fontSize: "12px", color: "#a1a1a6" }}>
                      {dim === "blur" ? "Sharpness (blur)" : dim === "exposure" ? "Exposure (lighting)" : dim === "content" ? "Content Detection" : dim}
                    </span>
                    <span style={{
                      fontSize: "12px", fontWeight: 600,
                      fontFamily: "SF Mono, Menlo, monospace",
                      color: passed ? "#34C759" : "#FF3B30",
                    }}>
                      {score.toFixed(2)} {passed ? "✓" : "✗"}
                    </span>
                  </div>
                  <div style={{ height: "6px", background: "var(--surface)", borderRadius: "3px", overflow: "hidden" }}>
                    <div style={{
                      height: "100%", borderRadius: "3px",
                      width: `${Math.min(score * 100, 100)}%`,
                      background: passed ? "#34C759" : "#FF3B30",
                      transition: "width 0.3s",
                    }} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* What the scores mean */}
          <div style={{ fontSize: "11px", fontWeight: 600, color: "#6E6E73", marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            What These Mean
          </div>
          <div style={{ fontSize: "11px", color: "#6E6E73", lineHeight: "1.6", marginBottom: "20px" }}>
            {Object.keys(selectedImg.scores).map((dim) => (
              <div key={dim} style={{ marginBottom: "4px" }}>
                <strong style={{ color: "var(--text-primary)" }}>
                  {dim === "blur" ? "Sharpness" : dim === "exposure" ? "Exposure" : dim === "content" ? "Content" : dim}:
                </strong>{" "}
                {dim === "blur" && "Measures focus quality using Laplacian variance. Higher = sharper."}
                {dim === "exposure" && "Measures lighting using histogram analysis. Mid-range (0.4-0.6) is ideal."}
                {dim === "content" && "Object detection confidence via NVIDIA GroundingDINO. Higher = more confident the target is present."}
                {!["blur", "exposure", "content"].includes(dim) && `Quality score for ${dim} dimension.`}
              </div>
            ))}
          </div>

          {/* Close button */}
          <button
            onClick={() => setSelectedImg(null)}
            style={{
              width: "100%",
              padding: "8px",
              background: "var(--surface)",
              border: "none",
              borderRadius: "8px",
              color: "var(--text-secondary)",
              fontSize: "13px",
              cursor: "pointer",
              fontFamily: "inherit",
            }}
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}

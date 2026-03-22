"use client";
import { useRef, useEffect } from "react";
import type { PipelineEvent } from "@/lib/types";

export default function ProgressLog({ events }: { events: PipelineEvent[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [events.length]);

  const displayEvents = events.filter((e) => e.type !== "ping");

  return (
    <div style={{
      background: "var(--surface)",
      borderRadius: "12px",
      padding: "14px 16px",
      maxHeight: "220px",
      overflowY: "auto",
      fontFamily: "SF Mono, Menlo, monospace",
      fontSize: "12px",
      marginBottom: "16px",
      lineHeight: "1.6",
    }}>
      {displayEvents.map((event, i) => (
        <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: "8px", padding: "2px 0" }}>
          <span style={{ color: getModuleColor(event), minWidth: "130px", flexShrink: 0 }}>
            [{event.module}]
          </span>
          <span style={{ color: getMessageColor(event), flex: 1 }}>
            {formatEvent(event)}
          </span>
          {(event as any).duration_seconds != null && event.type === "ModuleCompleted" && (
            <span style={{ color: "#6E6E73", fontSize: "11px", flexShrink: 0 }}>
              {(event as any).duration_seconds.toFixed(1)}s
            </span>
          )}
        </div>
      ))}
      {displayEvents.length === 0 && (
        <div style={{ color: "#6E6E73" }}>Waiting for pipeline events...</div>
      )}
      <div ref={endRef} />
    </div>
  );
}

function getModuleColor(event: PipelineEvent): string {
  if (event.type === "PipelineErrorEvent") return "#FF3B30";
  if (event.module === "executor") return "#00C7FF";
  return "#0071E3";
}

function getMessageColor(event: PipelineEvent): string {
  if (event.type === "PipelineErrorEvent") return "#FF3B30";
  if (event.type === "ImageVerdict") {
    const v = (event as any).verdict;
    if (v === "usable") return "#34C759";
    if (v === "recoverable") return "#FF9F0A";
    if (v === "unusable") return "#FF3B30";
  }
  if (event.type === "ToolProgress") {
    return (event as any).passed ? "#34C759" : "#FF3B30";
  }
  return "#a1a1a6";
}

function formatEvent(e: PipelineEvent): string {
  const a = e as any;
  switch (e.type) {
    case "ModuleStarted": {
      const names: Record<string, string> = {
        dataset_resolver: "Resolving dataset...",
        spec_generator: "Generating validation spec...",
        calibrator: "Calibrating thresholds...",
        planner: "Planning validation strategy...",
        compiler: "Compiling execution program...",
        executor: "Executing validation on images...",
        supervisor: "Running QA checks...",
        reporter: "Generating report...",
      };
      return names[e.module] || `Started ${a.details || ""}`;
    }
    case "ModuleCompleted":
      return `Done${a.summary ? ` — ${a.summary}` : ""}`;

    case "ToolProgress": {
      const tools: Record<string, string> = {
        laplacian_blur: "Sharpness (blur detection)",
        histogram_exposure: "Exposure (lighting analysis)",
        pixel_stats: "Information content",
        nvidia_grounding_dino: "Object detection (GroundingDINO)",
        roboflow_object_detection: "Object detection (Roboflow)",
        gpt4o_vision_semantic: "Semantic quality (GPT-4o Vision)",
      };
      const label = tools[a.tool_name] || a.tool_name;
      return `${label}: ${a.score?.toFixed(2)} — ${a.passed ? "passed ✓" : "failed ✗"}`;
    }

    case "ImageProgress":
      return `Processing ${a.current}/${a.total} — ${(a.image_path || "").split("/").pop()}`;

    case "ImageVerdict": {
      const scores = a.scores ? Object.entries(a.scores).map(([k, v]: [string, any]) => `${k}=${v.toFixed(2)}`).join(", ") : "";
      return `${(a.image_path || "").split("/").pop()} → ${a.verdict}${scores ? ` (${scores})` : ""}`;
    }

    case "SpecGenerated":
      return `Spec: "${a.spec_summary}"`;

    case "PlanGenerated":
      return `Plan: ${a.steps_count} steps across tiers ${JSON.stringify(a.tiers)}`;

    case "DatasetResolved":
      return `Downloaded ${a.image_count} images from ${a.source}`;

    case "PipelineErrorEvent":
      return `ERROR: ${a.message}`;

    default:
      return e.type;
  }
}

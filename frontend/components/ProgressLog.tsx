"use client";
import { useRef, useEffect } from "react";
import type { PipelineEvent } from "@/lib/types";

export default function ProgressLog({ events }: { events: PipelineEvent[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [events.length]);

  return (
    <div className="bg-[var(--surface)] rounded-[12px] p-4 max-h-[200px] overflow-y-auto font-mono text-[12px] mb-4">
      {events.map((event, i) => (
        <div key={i} className="flex items-center gap-2 py-[3px]">
          <span className="text-[var(--accent)] min-w-[130px]">[{event.module}]</span>
          <span className="text-[var(--text-secondary)] flex-1">{formatEvent(event)}</span>
          {event.type === "ModuleCompleted" && <span className="text-[var(--text-secondary)] text-[11px]">{(event as any).duration_seconds?.toFixed(1)}s</span>}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}

function formatEvent(e: PipelineEvent): string {
  switch (e.type) {
    case "ModuleStarted": return `Started ${(e as any).details || ""}`;
    case "ModuleCompleted": return `Completed ${(e as any).summary || ""}`;
    case "ImageProgress": return `${(e as any).current}/${(e as any).total} — ${((e as any).image_path || "").split("/").pop()}`;
    case "ImageVerdict": return `${((e as any).image_path || "").split("/").pop()} → ${(e as any).verdict}`;
    case "SpecGenerated": return (e as any).spec_summary;
    case "PlanGenerated": return `${(e as any).steps_count} steps across tiers ${JSON.stringify((e as any).tiers)}`;
    case "DatasetResolved": return `Downloaded ${(e as any).image_count} images from ${(e as any).source}`;
    case "PipelineErrorEvent": return `ERROR: ${(e as any).message}`;
    default: return e.type;
  }
}

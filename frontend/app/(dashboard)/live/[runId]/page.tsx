"use client";
import { useEffect, useState, useRef } from "react";
import { useParams } from "next/navigation";
import { connectToRun } from "@/lib/ws";
import { getRun, getRunImages } from "@/lib/api";
import type { PipelineEvent } from "@/lib/types";
import StatsBar from "@/components/StatsBar";
import ProgressLog from "@/components/ProgressLog";
import ImageGrid from "@/components/ImageGrid";
import ExportDropdown from "@/components/ExportDropdown";

interface Img { image_id: string; image_path?: string; verdict: string; scores: Record<string, number>; }

export default function LiveViewPage() {
  const params = useParams();
  const runId = params.runId as string;
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [images, setImages] = useState<Img[]>([]);
  const [stats, setStats] = useState({ processed: 0, total: 0, usable: 0, recoverable: 0, unusable: 0, elapsed: 0 });
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState("");
  const startTime = useRef(Date.now());

  useEffect(() => {
    const ws = connectToRun(runId, (event) => {
      setEvents((prev) => [...prev, event]);
      handleEvent(event);
    }, () => loadCompletedRun());

    const timer = setInterval(() => {
      if (!completed) setStats((prev) => ({ ...prev, elapsed: (Date.now() - startTime.current) / 1000 }));
    }, 100);

    return () => { ws.close(); clearInterval(timer); };
  }, [runId]);

  function handleEvent(event: PipelineEvent) {
    if (event.type === "ImageProgress") { const e = event as any; setStats((p) => ({ ...p, processed: e.current, total: e.total })); }
    if (event.type === "ImageVerdict") {
      const e = event as any;
      setImages((p) => [...p, { image_id: e.image_id, image_path: e.image_path, verdict: e.verdict, scores: e.scores || {} }]);
      setStats((p) => ({
        ...p,
        usable: p.usable + (e.verdict === "usable" ? 1 : 0),
        recoverable: p.recoverable + (e.verdict === "recoverable" ? 1 : 0),
        unusable: p.unusable + (e.verdict === "unusable" ? 1 : 0),
      }));
    }
    if (event.type === "RunCompleted") setCompleted(true);
    if (event.type === "ModuleCompleted" && (event as any).module === "executor") setStats((p) => ({ ...p, elapsed: (event as any).duration_seconds }));
  }

  async function loadCompletedRun() {
    try {
      const run = await getRun(runId);
      if (run.status === "completed" || run.status === "failed") {
        setCompleted(true);
        setStats({ processed: run.total_images || 0, total: run.total_images || 0, usable: run.usable_count || 0, recoverable: run.recoverable_count || 0, unusable: run.unusable_count || 0, elapsed: 0 });
        const imgs = await getRunImages(runId);
        setImages(imgs.map((img: any) => ({ image_id: img.image_id, image_path: img.image_path, verdict: img.verdict, scores: typeof img.scores === "string" ? JSON.parse(img.scores) : img.scores || {} })));
      }
    } catch { setError("Run not found"); }
  }

  const progress = stats.total > 0 ? (stats.processed / stats.total) * 100 : 0;

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-[22px] font-bold tracking-tight">
            Run {runId}
            {!completed && <span className="ml-2 text-[13px] font-normal text-[var(--accent)]">● Live</span>}
          </h2>
        </div>
        {completed && <ExportDropdown runId={runId} />}
      </div>
      <StatsBar {...stats} />
      <div className="h-[3px] bg-[var(--surface)] rounded-full mb-4 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${progress}%`, background: completed ? "#34C759" : "linear-gradient(90deg, #0071E3, #00C7FF)" }} />
      </div>
      {error && <div className="text-[#FF3B30] text-[13px] mb-4">{error}</div>}
      <ProgressLog events={events} />
      <ImageGrid images={images} total={stats.total || images.length} />
    </div>
  );
}

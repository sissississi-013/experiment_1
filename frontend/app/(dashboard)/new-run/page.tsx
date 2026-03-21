"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { createRun } from "@/lib/api";

export default function NewRunPage() {
  const router = useRouter();
  const [intent, setIntent] = useState("");
  const [datasetDesc, setDatasetDesc] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!intent.trim()) return;
    setLoading(true);
    setError("");
    try {
      const result = await createRun({ intent: intent.trim(), dataset_description: datasetDesc.trim() || intent.trim() });
      router.push(`/live/${result.run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start run");
      setLoading(false);
    }
  }

  return (
    <div className="p-8 max-w-[980px] mx-auto">
      <div className="mb-8">
        <h2 className="text-[28px] font-bold tracking-tight mb-1">Start a Validation Run</h2>
        <p className="text-[var(--text-secondary)] text-[15px]">Describe what you&apos;re looking for and the pipeline handles the rest.</p>
      </div>
      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        <div>
          <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-1.5">What do you want to validate?</label>
          <textarea className="input-apple min-h-[80px] resize-none text-[16px]" placeholder="find 10 sharp, well-exposed images of horses from COCO" value={intent} onChange={(e) => setIntent(e.target.value)} rows={3} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-1.5">Dataset Description <span className="font-normal">(optional)</span></label>
            <input className="input-apple" placeholder="Auto-detected from intent" value={datasetDesc} onChange={(e) => setDatasetDesc(e.target.value)} />
          </div>
          <div>
            <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-1.5">Max Images</label>
            <input className="input-apple" type="number" defaultValue="50" />
          </div>
        </div>
        {error && <div className="text-[#FF3B30] text-[13px] bg-[rgba(255,59,48,0.1)] rounded-[12px] p-3">{error}</div>}
        <div className="flex justify-end mt-2">
          <button type="submit" className="btn-primary px-8" disabled={loading || !intent.trim()}>
            {loading ? "Starting..." : "Start Validation"}
          </button>
        </div>
      </form>
    </div>
  );
}

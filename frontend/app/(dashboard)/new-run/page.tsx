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

  const canSubmit = intent.trim().length > 0 && !loading;

  async function handleClick() {
    if (!canSubmit) return;
    setLoading(true);
    setError("");
    try {
      const result = await createRun({
        intent: intent.trim(),
        dataset_description: datasetDesc.trim() || intent.trim(),
      });
      router.push(`/live/${result.run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start run");
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "32px", maxWidth: "980px", margin: "0 auto" }}>
      <div style={{ marginBottom: "32px" }}>
        <h2 style={{ fontSize: "28px", fontWeight: 700, letterSpacing: "-0.02em", marginBottom: "4px" }}>
          Start a Validation Run
        </h2>
        <p style={{ color: "#6E6E73", fontSize: "15px" }}>
          Describe what you&apos;re looking for and the pipeline handles the rest.
        </p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
        <div>
          <label style={{ display: "block", fontSize: "12px", fontWeight: 500, color: "#6E6E73", marginBottom: "6px" }}>
            What do you want to validate?
          </label>
          <textarea
            style={{
              width: "100%",
              minHeight: "80px",
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "12px",
              padding: "12px 14px",
              fontSize: "16px",
              color: "var(--text-primary)",
              fontFamily: "inherit",
              resize: "none",
            }}
            placeholder="find 10 sharp, well-exposed images of horses from COCO"
            value={intent}
            onChange={(e) => setIntent(e.target.value)}
            rows={3}
          />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
          <div>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 500, color: "#6E6E73", marginBottom: "6px" }}>
              Dataset Description <span style={{ fontWeight: 400 }}>(optional)</span>
            </label>
            <input
              style={{
                width: "100%",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: "12px",
                padding: "10px 14px",
                fontSize: "15px",
                color: "var(--text-primary)",
                fontFamily: "inherit",
                minHeight: "44px",
              }}
              placeholder="Auto-detected from intent"
              value={datasetDesc}
              onChange={(e) => setDatasetDesc(e.target.value)}
            />
          </div>
          <div>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 500, color: "#6E6E73", marginBottom: "6px" }}>
              Max Images
            </label>
            <input
              style={{
                width: "100%",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: "12px",
                padding: "10px 14px",
                fontSize: "15px",
                color: "var(--text-primary)",
                fontFamily: "inherit",
                minHeight: "44px",
              }}
              type="number"
              defaultValue="50"
            />
          </div>
        </div>

        {error && (
          <div style={{ color: "#FF3B30", fontSize: "13px", background: "rgba(255,59,48,0.1)", borderRadius: "12px", padding: "12px" }}>
            {error}
          </div>
        )}

        <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "8px" }}>
          <button
            type="button"
            onClick={handleClick}
            style={{
              background: canSubmit ? "#0071E3" : "rgba(0,113,227,0.4)",
              color: "white",
              border: "none",
              borderRadius: "980px",
              padding: "12px 32px",
              fontSize: "15px",
              fontWeight: 500,
              cursor: canSubmit ? "pointer" : "not-allowed",
              minHeight: "44px",
              fontFamily: "inherit",
              transition: "background 0.2s",
            }}
            onMouseEnter={(e) => { if (canSubmit) (e.target as HTMLButtonElement).style.background = "#0077ED"; }}
            onMouseLeave={(e) => { if (canSubmit) (e.target as HTMLButtonElement).style.background = "#0071E3"; }}
          >
            {loading ? "Starting..." : "Start Validation"}
          </button>
        </div>
      </div>
    </div>
  );
}

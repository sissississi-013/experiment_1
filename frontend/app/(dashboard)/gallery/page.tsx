"use client";
import { useEffect, useState } from "react";
import { listRuns, getRunImages, getImageUrl } from "@/lib/api";

interface ImageItem {
  image_id: string;
  image_path: string;
  verdict: string;
  scores: Record<string, number>;
  flags: string[];
  run_id?: string;
}

const verdictColors: Record<string, string> = {
  usable: "#34C759",
  recoverable: "#FF9F0A",
  unusable: "#FF3B30",
  error: "#FF453A",
};

export default function GalleryPage() {
  const [images, setImages] = useState<ImageItem[]>([]);
  const [selected, setSelected] = useState<ImageItem | null>(null);
  const [filter, setFilter] = useState<string>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadImages();
  }, []);

  async function loadImages() {
    try {
      const runs = await listRuns(10);
      const allImages: ImageItem[] = [];
      for (const run of runs) {
        if (run.status !== "completed") continue;
        try {
          const imgs = await getRunImages(run.id);
          for (const img of imgs) {
            allImages.push({
              ...img,
              scores: typeof img.scores === "string" ? JSON.parse(img.scores) : img.scores || {},
              flags: typeof img.flags === "string" ? JSON.parse(img.flags) : img.flags || [],
              run_id: run.id,
            });
          }
        } catch { /* skip failed runs */ }
      }
      setImages(allImages);
    } catch {
      setError("Failed to load images. Is the API server running?");
    } finally {
      setLoading(false);
    }
  }

  const filtered = filter === "all" ? images : images.filter((img) => img.verdict === filter);

  function basename(path: string) {
    return path?.split("/").pop() || path;
  }

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* Filter sidebar */}
      <div style={{
        width: "200px",
        padding: "20px 12px",
        borderRight: "1px solid var(--border)",
        flexShrink: 0,
      }}>
        <div style={{ fontSize: "11px", fontWeight: 600, color: "#6E6E73", textTransform: "uppercase", letterSpacing: "0.06em", padding: "6px 8px", marginBottom: "4px" }}>
          Filter by Verdict
        </div>
        {["all", "usable", "recoverable", "unusable", "error"].map((v) => (
          <div
            key={v}
            onClick={() => setFilter(v)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "6px 8px",
              borderRadius: "6px",
              fontSize: "13px",
              cursor: "pointer",
              background: filter === v ? "rgba(0,113,227,0.25)" : "transparent",
              color: filter === v ? "white" : "var(--text-primary)",
              marginBottom: "2px",
              transition: "background 0.2s",
            }}
          >
            {v !== "all" && (
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: verdictColors[v] || "#6E6E73" }} />
            )}
            {v.charAt(0).toUpperCase() + v.slice(1)}
            <span style={{ marginLeft: "auto", fontSize: "11px", color: "#6E6E73" }}>
              {v === "all" ? images.length : images.filter((i) => i.verdict === v).length}
            </span>
          </div>
        ))}
      </div>

      {/* Main content */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ padding: "24px 24px 16px" }}>
          <h2 style={{ fontSize: "22px", fontWeight: 700, letterSpacing: "-0.02em", marginBottom: "4px" }}>
            Image Gallery
          </h2>
          <p style={{ color: "#6E6E73", fontSize: "13px" }}>
            {filtered.length} images{filter !== "all" ? ` (${filter})` : ""} across all runs
          </p>
        </div>

        {loading && <p style={{ padding: "24px", color: "#6E6E73" }}>Loading...</p>}
        {error && <p style={{ padding: "24px", color: "#FF3B30" }}>{error}</p>}

        {!loading && filtered.length === 0 && !error && (
          <p style={{ padding: "24px", color: "#6E6E73" }}>No images found. Run a validation first.</p>
        )}

        <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
          {/* Image grid */}
          <div style={{
            flex: 1,
            overflowY: "auto",
            padding: "0 24px 24px",
          }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "4px" }}>
              {filtered.map((img, i) => (
                <div
                  key={`${img.run_id}-${img.image_id}-${i}`}
                  onClick={() => setSelected(img)}
                  style={{
                    aspectRatio: "1",
                    background: "var(--surface)",
                    borderRadius: "6px",
                    position: "relative",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    overflow: "hidden",
                    border: selected?.image_id === img.image_id && selected?.run_id === img.run_id
                      ? "2px solid #0071E3" : "2px solid transparent",
                    transition: "border-color 0.2s, transform 0.2s",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.02)")}
                  onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
                >
                  {img.image_path && (
                    <img
                      src={getImageUrl(img.image_path)}
                      alt={img.image_id}
                      style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "cover" }}
                      loading="lazy"
                    />
                  )}
                  <div style={{
                    position: "absolute", top: "6px", right: "6px",
                    width: "10px", height: "10px", borderRadius: "50%",
                    background: verdictColors[img.verdict] || "#6E6E73",
                    border: "1.5px solid rgba(0,0,0,0.3)",
                    zIndex: 2,
                  }} />
                  {!img.image_path && (
                    <span style={{ fontSize: "10px", fontFamily: "SF Mono, Menlo, monospace", color: "#6E6E73", zIndex: 1 }}>
                      {img.image_id.slice(-8)}
                    </span>
                  )}
                  <div style={{
                    position: "absolute", bottom: 0, left: 0, right: 0,
                    background: "linear-gradient(transparent, rgba(0,0,0,0.7))",
                    padding: "16px 6px 4px",
                    fontSize: "10px",
                    color: "white",
                    zIndex: 2,
                  }}>
                    {img.verdict}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Detail panel */}
          {selected && (
            <div style={{
              width: "320px",
              borderLeft: "1px solid var(--border)",
              padding: "24px",
              overflowY: "auto",
              flexShrink: 0,
            }}>
              <div style={{
                width: "100%",
                aspectRatio: "1",
                background: "var(--surface)",
                borderRadius: "12px",
                overflow: "hidden",
                marginBottom: "16px",
                position: "relative",
              }}>
                {selected.image_path ? (
                  <img
                    src={getImageUrl(selected.image_path)}
                    alt={selected.image_id}
                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  />
                ) : (
                  <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "12px", color: "#6E6E73" }}>
                    {basename(selected.image_path)}
                  </div>
                )}
              </div>

              <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "4px" }}>
                {basename(selected.image_path)}
              </h3>
              <p style={{
                fontSize: "13px",
                fontWeight: 600,
                color: verdictColors[selected.verdict] || "#6E6E73",
                marginBottom: "16px",
              }}>
                {selected.verdict.charAt(0).toUpperCase() + selected.verdict.slice(1)}
              </p>

              {/* Score bars */}
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {Object.entries(selected.scores).map(([dim, score]) => (
                  <div key={dim} style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <div style={{ width: "70px", fontSize: "12px", color: "#a1a1a6" }}>
                      {dim}
                    </div>
                    <div style={{
                      flex: 1, height: "6px", background: "var(--surface)",
                      borderRadius: "3px", overflow: "hidden",
                    }}>
                      <div style={{
                        height: "100%",
                        width: `${Math.min(score * 100, 100)}%`,
                        borderRadius: "3px",
                        background: score >= 0.5 ? "#34C759" : score >= 0.3 ? "#FF9F0A" : "#FF3B30",
                      }} />
                    </div>
                    <div style={{
                      width: "40px", textAlign: "right", fontSize: "12px",
                      fontWeight: 600, fontFamily: "SF Mono, Menlo, monospace",
                      color: score >= 0.5 ? "#34C759" : score >= 0.3 ? "#FF9F0A" : "#FF3B30",
                    }}>
                      {score.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>

              {/* Flags */}
              {selected.flags && selected.flags.length > 0 && (
                <div style={{ marginTop: "16px" }}>
                  <div style={{ fontSize: "11px", color: "#6E6E73", fontWeight: 500, marginBottom: "6px" }}>Flags</div>
                  <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                    {selected.flags.map((flag) => (
                      <span key={flag} style={{
                        fontSize: "11px", color: "#FF9F0A",
                        background: "rgba(255,159,10,0.1)",
                        padding: "2px 8px", borderRadius: "4px",
                      }}>
                        {flag}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Run ID */}
              {selected.run_id && (
                <div style={{ marginTop: "16px", fontSize: "11px", color: "#6E6E73" }}>
                  Run: <span style={{ color: "#0071E3", fontFamily: "SF Mono, Menlo, monospace" }}>{selected.run_id}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

import { getImageUrl } from "@/lib/api";

const vc: Record<string, string> = { usable: "#34C759", recoverable: "#FF9F0A", unusable: "#FF3B30", error: "#FF453A" };

interface Item { image_id: string; image_path?: string; verdict: string; scores: Record<string, number>; }

export default function ImageGrid({ images, total }: { images: Item[]; total: number }) {
  const slots = Array.from({ length: Math.max(total, images.length) }, (_, i) => images[i] || null);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "2px" }}>
      {slots.map((img, i) => (
        <div
          key={i}
          style={{
            aspectRatio: "1",
            background: "var(--surface)",
            borderRadius: "4px",
            position: "relative",
            overflow: "hidden",
            opacity: img ? 1 : 0.2,
          }}
        >
          {img && (
            <>
              {img.image_path ? (
                <img
                  src={getImageUrl(img.image_path)}
                  alt={img.image_id}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  loading="lazy"
                />
              ) : (
                <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "9px", color: "#6E6E73", fontFamily: "SF Mono, Menlo, monospace" }}>
                  {img.image_id.slice(-6)}
                </div>
              )}
              <div style={{
                position: "absolute", top: "6px", right: "6px",
                width: "10px", height: "10px", borderRadius: "50%",
                background: vc[img.verdict] || "#6E6E73",
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
                {Object.keys(img.scores).length > 0 && (
                  <span>{Object.values(img.scores)[0]?.toFixed(2)}</span>
                )}
              </div>
            </>
          )}
        </div>
      ))}
    </div>
  );
}

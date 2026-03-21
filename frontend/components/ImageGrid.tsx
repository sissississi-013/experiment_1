const vc: Record<string, string> = { usable: "#34C759", recoverable: "#FF9F0A", unusable: "#FF3B30", error: "#FF453A" };

interface Item { image_id: string; verdict: string; scores: Record<string, number>; }

export default function ImageGrid({ images, total }: { images: Item[]; total: number }) {
  const slots = Array.from({ length: Math.max(total, images.length) }, (_, i) => images[i] || null);
  return (
    <div className="grid grid-cols-5 gap-[2px]">
      {slots.map((img, i) => (
        <div key={i} className={`aspect-square bg-[var(--surface)] rounded relative flex items-center justify-center text-[10px] text-[var(--text-secondary)] ${!img ? "opacity-20" : ""}`}>
          {img && (
            <>
              <div className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full" style={{ background: vc[img.verdict] || "#6E6E73" }} />
              <span className="text-[9px] font-mono">{img.image_id.slice(-6)}</span>
              {Object.keys(img.scores).length > 0 && (
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-1 text-[9px] text-[#a1a1a6]">
                  {Object.values(img.scores)[0]?.toFixed(2)}
                </div>
              )}
            </>
          )}
        </div>
      ))}
    </div>
  );
}

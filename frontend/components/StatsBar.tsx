interface Props { processed: number; total: number; usable: number; recoverable: number; unusable: number; elapsed: number; }

export default function StatsBar({ processed, total, usable, recoverable, unusable, elapsed }: Props) {
  return (
    <div className="flex gap-[1px] mb-4">
      {[
        { label: "Processed", value: `${processed}/${total}`, color: "var(--text-primary)" },
        { label: "Usable", value: usable, color: "#34C759" },
        { label: "Recoverable", value: recoverable, color: "#FF9F0A" },
        { label: "Unusable", value: unusable, color: "#FF3B30" },
        { label: "Elapsed", value: `${elapsed.toFixed(1)}s`, color: "var(--text-primary)" },
      ].map((s) => (
        <div key={s.label} className="flex-1 bg-[var(--surface)] p-4 text-center first:rounded-l-[12px] last:rounded-r-[12px]">
          <div className="text-[26px] font-bold tracking-tight" style={{ color: s.color }}>{s.value}</div>
          <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">{s.label}</div>
        </div>
      ))}
    </div>
  );
}

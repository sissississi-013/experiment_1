"use client";
import { useState } from "react";
import { getExportUrl } from "@/lib/api";

export default function ExportDropdown({ runId }: { runId: string }) {
  const [open, setOpen] = useState(false);
  const opts = [
    { label: "Usable Only", filter: "usable" },
    { label: "Usable + Recoverable", filter: "usable+recoverable" },
    { label: "All Images", filter: "all" },
  ];
  return (
    <div className="relative inline-block">
      <button className="btn-primary px-6 text-[14px]" onClick={() => setOpen(!open)}>Export Dataset ↓</button>
      {open && (
        <div className="absolute right-0 mt-2 w-56 bg-[var(--elevated)] rounded-[12px] border border-[var(--border)] shadow-lg z-10 overflow-hidden">
          {opts.map((o) => (
            <a key={o.filter} href={getExportUrl(runId, o.filter)} className="block px-4 py-3 text-[13px] hover:bg-[var(--surface)] transition-colors" onClick={() => setOpen(false)}>{o.label}</a>
          ))}
        </div>
      )}
    </div>
  );
}

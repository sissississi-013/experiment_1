"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/new-run", label: "New Run", icon: "+" },
  { href: "/live", label: "Live View", icon: "▶" },
  { href: "/history", label: "History", icon: "☰" },
  { href: "/gallery", label: "Gallery", icon: "▣" },
];

export default function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="w-[240px] h-screen glass border-r border-[var(--glass-border)] flex flex-col p-3 flex-shrink-0 fixed left-0 top-0">
      <div className="px-2 py-3 mb-4">
        <h1 className="text-[15px] font-semibold tracking-tight">Validation Pipeline</h1>
      </div>
      <nav className="flex flex-col gap-0.5">
        <div className="px-2 py-1 text-[11px] font-semibold text-[var(--text-secondary)] uppercase tracking-widest">Navigation</div>
        {navItems.map((item) => {
          const isActive = pathname === item.href || pathname?.startsWith(item.href + "/");
          return (
            <Link key={item.href} href={item.href}
              className={`flex items-center gap-2 px-2 py-[6px] rounded-md text-[13px] transition-colors duration-200 ${
                isActive ? "bg-[rgba(0,113,227,0.25)] text-white" : "text-[var(--text-primary)] hover:bg-[rgba(255,255,255,0.06)]"
              }`}>
              <span className="w-5 text-center text-sm">{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}

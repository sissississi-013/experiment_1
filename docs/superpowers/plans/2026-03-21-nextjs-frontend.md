# Next.js Frontend Implementation Plan (Sub-project 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a macOS-native-feeling web console with Next.js that connects to the FastAPI backend — New Run form, live execution view with WebSocket streaming, and Apple's design system.

**Architecture:** Next.js 14 App Router with TypeScript. Tailwind CSS configured with Apple design tokens. Client components for real-time features (WebSocket). Server components for static layouts. Sidebar navigation persistent across all pages.

**Tech Stack:** Next.js 14, TypeScript, Tailwind CSS, React 18

**Spec:** `docs/superpowers/specs/2026-03-21-frontend-design.md`

---

### Task 1: Next.js scaffold + Apple design system

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.js`
- Create: `frontend/next.config.js`
- Create: `frontend/app/globals.css`
- Create: `frontend/app/layout.tsx`
- Create: `frontend/app/page.tsx`

- [ ] **Step 1: Initialize Next.js project**

Run from project root:
```bash
cd /Users/sissi/Desktop/validation-pipeline && npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir=false --import-alias="@/*" --no-turbopack
```

When prompted, accept all defaults.

- [ ] **Step 2: Configure Tailwind with Apple design tokens**

Replace `frontend/tailwind.config.ts`:
```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      fontFamily: {
        display: ["-apple-system", "BlinkMacSystemFont", "SF Pro Display", "Inter", "sans-serif"],
        body: ["-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Inter", "sans-serif"],
        mono: ["SF Mono", "Menlo", "monospace"],
      },
      colors: {
        apple: {
          bg: { light: "#FFFFFF", dark: "#000000" },
          surface: { light: "#F5F5F7", dark: "#1C1C1E" },
          elevated: { light: "#FFFFFF", dark: "#2C2C2E" },
          text: { primary: { light: "#1D1D1F", dark: "#F5F5F7" }, secondary: "#6E6E73" },
          accent: { DEFAULT: "#0071E3", hover: "#0077ED" },
          success: "#34C759",
          warning: "#FF9F0A",
          error: "#FF3B30",
          border: { light: "#D2D2D7", dark: "rgba(255,255,255,0.1)" },
        },
      },
      borderRadius: {
        apple: "12px",
        pill: "980px",
      },
      spacing: {
        "section": "80px",
        "hero": "120px",
      },
      maxWidth: {
        "content": "980px",
        "wide": "1440px",
      },
      transitionTimingFunction: {
        apple: "cubic-bezier(0.25, 0.1, 0.25, 1)",
      },
    },
  },
  plugins: [],
};
export default config;
```

- [ ] **Step 3: Create Apple design globals.css**

Replace `frontend/app/globals.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --bg: #FFFFFF;
  --surface: #F5F5F7;
  --elevated: #FFFFFF;
  --text-primary: #1D1D1F;
  --text-secondary: #6E6E73;
  --accent: #0071E3;
  --accent-hover: #0077ED;
  --border: #D2D2D7;
  --glass: rgba(255, 255, 255, 0.72);
  --glass-border: rgba(0, 0, 0, 0.06);
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #000000;
    --surface: #1C1C1E;
    --elevated: #2C2C2E;
    --text-primary: #F5F5F7;
    --text-secondary: #6E6E73;
    --border: rgba(255, 255, 255, 0.1);
    --glass: rgba(29, 29, 31, 0.72);
    --glass-border: rgba(255, 255, 255, 0.06);
  }
}

body {
  background: var(--bg);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif;
  font-size: 15px;
  line-height: 1.47;
  -webkit-font-smoothing: antialiased;
}

h1, h2, h3, h4 {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.2;
}

/* Frosted glass effect */
.glass {
  background: var(--glass);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid var(--glass-border);
}

/* Apple pill button */
.btn-primary {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 980px;
  padding: 10px 24px;
  font-size: 15px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
  min-height: 44px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.btn-primary:hover { background: var(--accent-hover); }

/* Apple input */
.input-apple {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 15px;
  color: var(--text-primary);
  width: 100%;
  min-height: 44px;
  font-family: inherit;
  transition: border-color 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
}
.input-apple:focus {
  outline: none;
  border-color: var(--accent);
}

/* Verdict colors */
.verdict-usable { color: #34C759; }
.verdict-recoverable { color: #FF9F0A; }
.verdict-unusable { color: #FF3B30; }
.verdict-error { color: #FF453A; }
```

- [ ] **Step 4: Create root layout**

Replace `frontend/app/layout.tsx`:
```tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Validation Pipeline",
  description: "Autonomous image dataset validation console",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
```

- [ ] **Step 5: Create redirect page**

Replace `frontend/app/page.tsx`:
```tsx
import { redirect } from "next/navigation";

export default function Home() {
  redirect("/new-run");
}
```

- [ ] **Step 6: Verify it builds**

Run:
```bash
cd /Users/sissi/Desktop/validation-pipeline/frontend && npm run build
```
Expected: Build succeeds (may warn about redirect, that's fine)

- [ ] **Step 7: Commit**

```bash
cd /Users/sissi/Desktop/validation-pipeline
echo "node_modules/" >> .gitignore
echo ".next/" >> .gitignore
git add frontend/ .gitignore
git commit -m "feat: Next.js scaffold with Apple design system (Tailwind + globals.css)"
```

---

### Task 2: Sidebar + app shell layout

**Files:**
- Create: `frontend/components/Sidebar.tsx`
- Create: `frontend/app/(dashboard)/layout.tsx`
- Create: `frontend/app/(dashboard)/new-run/page.tsx` (placeholder)

- [ ] **Step 1: Create Sidebar component**

```tsx
// frontend/components/Sidebar.tsx
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
        <h1 className="text-[15px] font-semibold tracking-tight">
          Validation Pipeline
        </h1>
      </div>

      <nav className="flex flex-col gap-0.5">
        <div className="px-2 py-1 text-[11px] font-semibold text-[var(--text-secondary)] uppercase tracking-widest">
          Navigation
        </div>
        {navItems.map((item) => {
          const isActive =
            pathname === item.href || pathname?.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2 px-2 py-[6px] rounded-md text-[13px] transition-colors duration-200 ease-apple ${
                isActive
                  ? "bg-[rgba(0,113,227,0.25)] text-white"
                  : "text-[var(--text-primary)] hover:bg-[rgba(255,255,255,0.06)]"
              }`}
            >
              <span className="w-5 text-center text-sm">{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
```

- [ ] **Step 2: Create dashboard layout with sidebar**

```tsx
// frontend/app/(dashboard)/layout.tsx
import Sidebar from "@/components/Sidebar";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 ml-[240px] min-h-screen">
        {children}
      </main>
    </div>
  );
}
```

- [ ] **Step 3: Create New Run placeholder page**

```tsx
// frontend/app/(dashboard)/new-run/page.tsx
export default function NewRunPage() {
  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold tracking-tight mb-2">
        Start a Validation Run
      </h2>
      <p className="text-[var(--text-secondary)]">
        Describe what you&apos;re looking for and the pipeline handles the rest.
      </p>
    </div>
  );
}
```

- [ ] **Step 4: Update root page redirect**

Update `frontend/app/page.tsx` to redirect to `/new-run`.

- [ ] **Step 5: Verify dev server**

Run:
```bash
cd /Users/sissi/Desktop/validation-pipeline/frontend && npm run dev
```
Open http://localhost:3000 — should see sidebar + "Start a Validation Run" heading.

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "feat: add Sidebar component and dashboard shell layout"
```

---

### Task 3: API client + TypeScript types

**Files:**
- Create: `frontend/lib/api.ts`
- Create: `frontend/lib/ws.ts`
- Create: `frontend/lib/types.ts`

- [ ] **Step 1: Create TypeScript types matching Pydantic schemas**

```tsx
// frontend/lib/types.ts
export interface Run {
  id: string;
  intent: string;
  status: "running" | "completed" | "failed";
  dataset_path?: string;
  dataset_description?: string;
  total_images?: number;
  usable_count?: number;
  recoverable_count?: number;
  unusable_count?: number;
  error_count?: number;
  overall_score?: number;
  created_at?: string;
  completed_at?: string;
}

export interface ImageResult {
  image_id: string;
  image_path: string;
  verdict: "usable" | "recoverable" | "unusable" | "error";
  scores: Record<string, number>;
  errors: string[];
  flags: string[];
}

export interface CreateRunRequest {
  intent: string;
  dataset_path?: string;
  dataset_description?: string;
}

export interface PipelineEvent {
  type: string;
  module: string;
  timestamp: string;
  [key: string]: unknown;
}

export interface ModuleStarted extends PipelineEvent {
  type: "ModuleStarted";
  details: string;
}

export interface ModuleCompleted extends PipelineEvent {
  type: "ModuleCompleted";
  duration_seconds: number;
  summary: string;
}

export interface ImageProgress extends PipelineEvent {
  type: "ImageProgress";
  current: number;
  total: number;
  image_path: string;
}

export interface ImageVerdict extends PipelineEvent {
  type: "ImageVerdict";
  image_id: string;
  image_path: string;
  verdict: string;
  scores: Record<string, number>;
  errors: string[];
}

export interface SpecGenerated extends PipelineEvent {
  type: "SpecGenerated";
  spec_summary: string;
  quality_criteria: string[];
  content_criteria: string[];
}

export interface PlanGenerated extends PipelineEvent {
  type: "PlanGenerated";
  steps_count: number;
  tiers: number[];
}
```

- [ ] **Step 2: Create REST API client**

```tsx
// frontend/lib/api.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function listRuns(limit = 20): Promise<any[]> {
  return fetchAPI(`/api/runs?limit=${limit}`);
}

export async function getRun(runId: string): Promise<any> {
  return fetchAPI(`/api/runs/${runId}`);
}

export async function createRun(body: {
  intent: string;
  dataset_path?: string;
  dataset_description?: string;
}): Promise<{ run_id: string; status: string }> {
  return fetchAPI("/api/runs", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getRunImages(
  runId: string,
  verdict?: string
): Promise<any[]> {
  const params = verdict ? `?verdict=${verdict}` : "";
  return fetchAPI(`/api/runs/${runId}/images${params}`);
}

export function getExportUrl(runId: string, filter = "usable"): string {
  return `${API_BASE}/api/runs/${runId}/export?filter=${filter}`;
}
```

- [ ] **Step 3: Create WebSocket client**

```tsx
// frontend/lib/ws.ts
import type { PipelineEvent } from "./types";

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export function connectToRun(
  runId: string,
  onEvent: (event: PipelineEvent) => void,
  onClose?: () => void
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/api/runs/${runId}/stream`);

  ws.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data) as PipelineEvent;
      if (event.type !== "ping") {
        onEvent(event);
      }
    } catch {
      // ignore malformed messages
    }
  };

  ws.onclose = () => {
    onClose?.();
  };

  ws.onerror = () => {
    onClose?.();
  };

  return ws;
}
```

- [ ] **Step 4: Create .env.local for frontend**

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/ frontend/.env.local
git commit -m "feat: add API client, WebSocket client, and TypeScript types"
```

---

### Task 4: New Run page — form + submit

**Files:**
- Create: `frontend/app/(dashboard)/new-run/page.tsx` (replace placeholder)

- [ ] **Step 1: Implement New Run page**

```tsx
// frontend/app/(dashboard)/new-run/page.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createRun } from "@/lib/api";

export default function NewRunPage() {
  const router = useRouter();
  const [intent, setIntent] = useState("");
  const [datasetDesc, setDatasetDesc] = useState("");
  const [maxImages, setMaxImages] = useState("50");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!intent.trim()) return;

    setLoading(true);
    setError("");

    try {
      const description = datasetDesc.trim() || intent;
      const result = await createRun({
        intent: intent.trim(),
        dataset_description: description,
      });
      router.push(`/live/${result.run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start run");
      setLoading(false);
    }
  }

  return (
    <div className="p-8 max-w-content mx-auto">
      <div className="mb-8">
        <h2 className="text-[28px] font-bold tracking-tight mb-1">
          Start a Validation Run
        </h2>
        <p className="text-[var(--text-secondary)] text-[15px]">
          Describe what you&apos;re looking for and the pipeline handles the rest.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        <div>
          <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-1.5">
            What do you want to validate?
          </label>
          <textarea
            className="input-apple min-h-[80px] resize-none text-[16px]"
            placeholder="find 10 sharp, well-exposed images of horses from COCO"
            value={intent}
            onChange={(e) => setIntent(e.target.value)}
            rows={3}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-1.5">
              Dataset Description
              <span className="text-[var(--text-secondary)] font-normal"> (optional)</span>
            </label>
            <input
              className="input-apple"
              placeholder="Auto-detected from intent"
              value={datasetDesc}
              onChange={(e) => setDatasetDesc(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-1.5">
              Max Images
            </label>
            <input
              className="input-apple"
              type="number"
              value={maxImages}
              onChange={(e) => setMaxImages(e.target.value)}
            />
          </div>
        </div>

        {error && (
          <div className="text-[#FF3B30] text-[13px] bg-[rgba(255,59,48,0.1)] rounded-apple p-3">
            {error}
          </div>
        )}

        <div className="flex justify-end mt-2">
          <button
            type="submit"
            className="btn-primary px-8 text-[15px]"
            disabled={loading || !intent.trim()}
          >
            {loading ? "Starting..." : "Start Validation"}
          </button>
        </div>
      </form>
    </div>
  );
}
```

- [ ] **Step 2: Verify in browser**

Run frontend dev server (`npm run dev`), open http://localhost:3000/new-run. Should see the form with Apple styling.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/
git commit -m "feat: add New Run page with intent form and API submit"
```

---

### Task 5: Live View page — WebSocket + real-time UI

**Files:**
- Create: `frontend/components/StatsBar.tsx`
- Create: `frontend/components/ProgressLog.tsx`
- Create: `frontend/components/ImageGrid.tsx`
- Create: `frontend/components/ExportDropdown.tsx`
- Create: `frontend/app/(dashboard)/live/[runId]/page.tsx`

- [ ] **Step 1: Create StatsBar component**

```tsx
// frontend/components/StatsBar.tsx
interface StatsBarProps {
  processed: number;
  total: number;
  usable: number;
  recoverable: number;
  unusable: number;
  elapsed: number;
}

export default function StatsBar({ processed, total, usable, recoverable, unusable, elapsed }: StatsBarProps) {
  return (
    <div className="flex gap-[1px] mb-4">
      {[
        { label: "Processed", value: `${processed}/${total}`, color: "var(--text-primary)" },
        { label: "Usable", value: usable, color: "#34C759" },
        { label: "Recoverable", value: recoverable, color: "#FF9F0A" },
        { label: "Unusable", value: unusable, color: "#FF3B30" },
        { label: "Elapsed", value: `${elapsed.toFixed(1)}s`, color: "var(--text-primary)" },
      ].map((stat) => (
        <div key={stat.label} className="flex-1 bg-[var(--surface)] p-4 text-center first:rounded-l-apple last:rounded-r-apple">
          <div className="text-[26px] font-bold tracking-tight" style={{ color: stat.color }}>
            {stat.value}
          </div>
          <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">{stat.label}</div>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Create ProgressLog component**

```tsx
// frontend/components/ProgressLog.tsx
"use client";

import { useRef, useEffect } from "react";
import type { PipelineEvent } from "@/lib/types";

interface ProgressLogProps {
  events: PipelineEvent[];
}

export default function ProgressLog({ events }: ProgressLogProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  return (
    <div className="bg-[var(--surface)] rounded-apple p-4 max-h-[200px] overflow-y-auto font-mono text-[12px] mb-4">
      {events.map((event, i) => (
        <div key={i} className="flex items-center gap-2 py-[3px]">
          <span className="text-[var(--accent)] min-w-[130px]">[{event.module}]</span>
          <span className="text-[var(--text-secondary)] flex-1">
            {formatEvent(event)}
          </span>
          {event.type === "ModuleCompleted" && (
            <span className="text-[var(--text-secondary)] text-[11px]">
              {(event as any).duration_seconds?.toFixed(1)}s
            </span>
          )}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}

function formatEvent(event: PipelineEvent): string {
  switch (event.type) {
    case "ModuleStarted": return `Started ${(event as any).details || ""}`;
    case "ModuleCompleted": return `Completed ${(event as any).summary || ""}`;
    case "ImageProgress": return `${(event as any).current}/${(event as any).total} — ${basename((event as any).image_path)}`;
    case "ImageVerdict": return `${basename((event as any).image_path)} → ${(event as any).verdict}`;
    case "SpecGenerated": return (event as any).spec_summary;
    case "PlanGenerated": return `${(event as any).steps_count} steps across tiers ${JSON.stringify((event as any).tiers)}`;
    case "DatasetResolved": return `Downloaded ${(event as any).image_count} images from ${(event as any).source}`;
    case "PipelineErrorEvent": return `ERROR: ${(event as any).message}`;
    default: return event.type;
  }
}

function basename(path: string): string {
  return path?.split("/").pop() || path;
}
```

- [ ] **Step 3: Create ImageGrid component**

```tsx
// frontend/components/ImageGrid.tsx
interface ImageGridItem {
  image_id: string;
  verdict: string;
  scores: Record<string, number>;
}

interface ImageGridProps {
  images: ImageGridItem[];
  total: number;
}

const verdictColor: Record<string, string> = {
  usable: "#34C759",
  recoverable: "#FF9F0A",
  unusable: "#FF3B30",
  error: "#FF453A",
};

export default function ImageGrid({ images, total }: ImageGridProps) {
  const slots = Array.from({ length: total }, (_, i) => images[i] || null);

  return (
    <div className="grid grid-cols-5 gap-[2px]">
      {slots.map((img, i) => (
        <div
          key={i}
          className={`aspect-square bg-[var(--surface)] rounded relative flex items-center justify-center text-[10px] text-[var(--text-secondary)] transition-opacity duration-300 ${
            !img ? "opacity-20" : ""
          }`}
        >
          {img && (
            <>
              <div
                className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full"
                style={{ background: verdictColor[img.verdict] || "#6E6E73" }}
              />
              <span className="text-[9px] font-mono">{img.image_id.slice(-6)}</span>
              {img.scores && Object.keys(img.scores).length > 0 && (
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
```

- [ ] **Step 4: Create ExportDropdown component**

```tsx
// frontend/components/ExportDropdown.tsx
"use client";

import { useState } from "react";
import { getExportUrl } from "@/lib/api";

interface ExportDropdownProps {
  runId: string;
}

export default function ExportDropdown({ runId }: ExportDropdownProps) {
  const [open, setOpen] = useState(false);

  const options = [
    { label: "Usable Only", filter: "usable" },
    { label: "Usable + Recoverable", filter: "usable+recoverable" },
    { label: "All Images", filter: "all" },
  ];

  return (
    <div className="relative inline-block">
      <button
        className="btn-primary px-6 text-[14px]"
        onClick={() => setOpen(!open)}
      >
        Export Dataset ↓
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-56 bg-[var(--elevated)] rounded-apple border border-[var(--border)] shadow-lg z-10 overflow-hidden">
          {options.map((opt) => (
            <a
              key={opt.filter}
              href={getExportUrl(runId, opt.filter)}
              className="block px-4 py-3 text-[13px] hover:bg-[var(--surface)] transition-colors"
              onClick={() => setOpen(false)}
            >
              {opt.label}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Create Live View page**

```tsx
// frontend/app/(dashboard)/live/[runId]/page.tsx
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

interface ImageItem {
  image_id: string;
  verdict: string;
  scores: Record<string, number>;
}

export default function LiveViewPage() {
  const params = useParams();
  const runId = params.runId as string;

  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [stats, setStats] = useState({
    processed: 0, total: 0, usable: 0, recoverable: 0, unusable: 0, elapsed: 0,
  });
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState("");
  const startTime = useRef(Date.now());

  useEffect(() => {
    // Try connecting via WebSocket for live run
    const ws = connectToRun(
      runId,
      (event) => {
        setEvents((prev) => [...prev, event]);
        handleEvent(event);
      },
      () => {
        // Connection closed — might be a completed run, load from API
        loadCompletedRun();
      }
    );

    // Update elapsed time
    const timer = setInterval(() => {
      if (!completed) {
        setStats((prev) => ({
          ...prev,
          elapsed: (Date.now() - startTime.current) / 1000,
        }));
      }
    }, 100);

    return () => {
      ws.close();
      clearInterval(timer);
    };
  }, [runId]);

  function handleEvent(event: PipelineEvent) {
    if (event.type === "ImageProgress") {
      const e = event as any;
      setStats((prev) => ({ ...prev, processed: e.current, total: e.total }));
    }
    if (event.type === "ImageVerdict") {
      const e = event as any;
      setImages((prev) => [
        ...prev,
        { image_id: e.image_id, verdict: e.verdict, scores: e.scores || {} },
      ]);
      setStats((prev) => ({
        ...prev,
        usable: prev.usable + (e.verdict === "usable" ? 1 : 0),
        recoverable: prev.recoverable + (e.verdict === "recoverable" ? 1 : 0),
        unusable: prev.unusable + (e.verdict === "unusable" ? 1 : 0),
      }));
    }
    if (event.type === "RunCompleted") {
      setCompleted(true);
    }
    if (event.type === "ModuleCompleted") {
      const e = event as any;
      if (e.module === "executor") {
        setStats((prev) => ({ ...prev, elapsed: e.duration_seconds }));
      }
    }
  }

  async function loadCompletedRun() {
    try {
      const run = await getRun(runId);
      if (run.status === "completed" || run.status === "failed") {
        setCompleted(true);
        setStats({
          processed: run.total_images || 0,
          total: run.total_images || 0,
          usable: run.usable_count || 0,
          recoverable: run.recoverable_count || 0,
          unusable: run.unusable_count || 0,
          elapsed: 0,
        });
        const imgs = await getRunImages(runId);
        setImages(
          imgs.map((img: any) => ({
            image_id: img.image_id,
            verdict: img.verdict,
            scores: typeof img.scores === "string" ? JSON.parse(img.scores) : img.scores || {},
          }))
        );
      }
    } catch {
      setError("Run not found");
    }
  }

  const progress =
    stats.total > 0 ? (stats.processed / stats.total) * 100 : 0;

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-[22px] font-bold tracking-tight">
            Run {runId}
            {!completed && (
              <span className="ml-2 text-[13px] font-normal text-[var(--accent)]">
                ● Live
              </span>
            )}
          </h2>
        </div>
        {completed && <ExportDropdown runId={runId} />}
      </div>

      <StatsBar {...stats} />

      {/* Progress bar */}
      <div className="h-[3px] bg-[var(--surface)] rounded-full mb-4 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${progress}%`,
            background: completed
              ? "#34C759"
              : "linear-gradient(90deg, #0071E3, #00C7FF)",
          }}
        />
      </div>

      {error && (
        <div className="text-[#FF3B30] text-[13px] mb-4">{error}</div>
      )}

      <ProgressLog events={events} />

      <ImageGrid images={images} total={stats.total || images.length} />
    </div>
  );
}
```

- [ ] **Step 6: Verify in browser**

Start both servers:
- Terminal 1: `cd /Users/sissi/Desktop/validation-pipeline && python3 api/run_dev.py`
- Terminal 2: `cd /Users/sissi/Desktop/validation-pipeline/frontend && npm run dev`

Open http://localhost:3000/new-run, type an intent, click "Start Validation". Should redirect to live view and show real-time progress.

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "feat: add Live View page with WebSocket streaming, StatsBar, ProgressLog, ImageGrid, ExportDropdown"
```

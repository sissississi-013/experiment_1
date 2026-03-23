# Frontend — Validation Pipeline Console

**Date**: 2026-03-21
**Status**: Draft

## Overview

A macOS-native-feeling web console for the validation pipeline. Built with Next.js + FastAPI, following Apple's iOS 18 / macOS Sequoia design system. Real-time pipeline execution via WebSocket, persistent run history from Neon, and dataset export as ZIP.

## Why

The pipeline currently runs from the CLI. For Apple researchers, a polished visual console makes the system accessible, transparent, and production-ready. Every pipeline step should be observable in real-time, results should be browsable, and curated datasets should be downloadable with one click.

---

## 1. Architecture

Three processes:

- **Next.js** (port 3000) — UI. App Router, TypeScript, Tailwind with Apple design tokens.
- **FastAPI** (port 8000) — API. REST for CRUD, WebSocket for live streaming. Wraps existing pipeline + EventBus + RunStore.
- **Neon Postgres** — Already built. Stores runs, events, image results.

```
Next.js ──REST──> FastAPI ──HTTP /sql──> Neon
Next.js <──WS──── FastAPI <──EventBus── Pipeline
```

### FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/runs | Start a new pipeline run (async background task) |
| GET | /api/runs | List runs (paginated) |
| GET | /api/runs/{id} | Get run detail with full report |
| GET | /api/runs/{id}/images | Get per-image results (filterable by verdict) |
| GET | /api/runs/{id}/export | Download curated dataset as ZIP |
| GET | /api/images | Query images across all runs |
| WS | /api/runs/{id}/stream | Live event stream for a running pipeline |

### Export Endpoint Detail

`GET /api/runs/{id}/export?filter=usable`

Query params:
- `filter`: `usable` (default), `usable+recoverable`, `all`
- `include_manifest`: `true` (default) — includes `manifest.json` with per-image metadata
- `include_report`: `false` (default) — includes full `report.json`

Returns: `application/zip` with images + manifest.json

### WebSocket Protocol

Client connects to `WS /api/runs/{id}/stream`. Server pushes JSON events matching the existing `PipelineEvent` Pydantic models:

```json
{"type": "ModuleStarted", "module": "executor", "details": "processing 10 images", "timestamp": "..."}
{"type": "ImageProgress", "module": "executor", "current": 3, "total": 10, "image_path": "..."}
{"type": "ImageVerdict", "module": "executor", "image_id": "000000319721", "verdict": "usable", "scores": {"blur": 1.0, "content": 0.61}}
{"type": "ModuleCompleted", "module": "executor", "duration_seconds": 33.5}
```

When the pipeline completes, server sends a final `{"type": "RunCompleted", "run_id": "..."}` and closes the connection.

---

## 2. Pages

### 2.1 New Run (Default Landing — `/new-run`)

Clean form:
- **Intent** input (large textarea, placeholder: "find 10 sharp, well-exposed images of horses from COCO")
- **Dataset Source** (auto-detected from intent, or manual: COCO / HuggingFace / URL / Local path)
- **Max Images** (number input, default 50)
- **Good/Bad Exemplars** (drag-drop zones, optional)
- **Start Validation** button (pill-shaped, #0071E3)

On submit: POST /api/runs → redirect to `/live/{runId}`

### 2.2 Live Execution (`/live/[runId]`)

Auto-switches here when a run starts. Shows:
- **Stats bar** — Processed/Total, Usable, Recoverable, Unusable, Elapsed time. Updates live.
- **Progress bar** — Thin gradient bar (blue→cyan) showing % complete
- **Event log** — Monospace streaming log of pipeline events. Auto-scrolls. Color-coded by module.
- **Image grid** — Thumbnails fill in as images are processed. Verdict dots (green/orange/red). Scores overlay.

When run completes:
- Stats bar shows final numbers
- **"Export Dataset" button** appears prominently
- Progress bar fills to 100% with a completion animation

### 2.3 Run History (`/history`)

Table of all runs from Neon:
- Columns: Run ID, Intent, Status (completed/failed), Images, Score, Time
- Click a row → navigate to `/live/{runId}` (shows completed results, not live)
- Status badges: green for completed, red for failed
- Sortable by any column

### 2.4 Image Gallery (`/gallery`)

Browse images across all runs:
- **Sidebar filters** — Verdict (usable/recoverable/unusable), run ID, score threshold
- **Image grid** — Large thumbnails with verdict dots
- **Detail panel** — Click an image to see:
  - Full image preview
  - Verdict + reason
  - Score bars per dimension (blur, exposure, content) with visual fill
  - Audit trail (tool name, raw value, calibrated value, threshold, pass/fail)
- **Bulk export** — Select images by filter, download as ZIP

---

## 3. Apple Design System Implementation

All design tokens from the user's specification implemented in Tailwind config + globals.css:

### Typography
- SF Pro Display for headings (bold 700, letter-spacing -0.02em to -0.04em)
- SF Pro Text for body (400 weight, 14-16px desktop)
- Fallback: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif
- Line height: 1.2 headings, 1.47 body

### Colors (Dark Mode Primary)
- Background: #000000
- Surface: #1C1C1E (cards), #2C2C2E (elevated)
- Primary text: #F5F5F7
- Secondary text: #6E6E73
- Accent: #0071E3 (hover #0077ED)
- Success: #34C759, Warning: #FF9F0A, Error: #FF3B30
- Frosted glass: rgba(29,29,31,0.72) + backdrop-filter: blur(20px) saturate(180%)

### Colors (Light Mode)
- Background: #FFFFFF
- Section backgrounds: #F5F5F7 or #FBFBFD
- Primary text: #1D1D1F
- Cards: rgba(255,255,255,0.72) + backdrop-filter: blur(20px) saturate(180%)

### Components
- Buttons: pill-shaped (border-radius: 980px), 44px min height, 17px font
- Inputs: 12px border-radius, 1px border #D2D2D7, 44px height
- Cards: 12-16px border-radius, 20-24px padding
- Sidebar: frosted glass, 240px width

### Motion
- Hover: transition all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1)
- Scroll fade-in: translateY(20px)→0, opacity 0→1, 0.6s
- Card hover: scale(1.02)
- Dark mode: prefers-color-scheme media query

### Layout
- Max content width: 980px text, 1440px edge-to-edge
- Vertical padding: 80px+ between sections
- CSS grid: 12 columns, 20px gap

---

## 4. User Flow

1. **Open app** → New Run page (default)
2. **Type intent** → Configure optional fields → Click "Start Validation"
3. **Auto-redirect to Live View** → Watch real-time progress (WebSocket)
4. **Run completes** → "Export Dataset" button appears → Choose export filter → Download ZIP
5. **Browse History** → Click any past run to view results → Re-export anytime
6. **Gallery** → Cross-run image browsing → Filter by verdict/score → Bulk export

---

## 5. File Structure

```
frontend/                         # Next.js app
    app/
        layout.tsx                # Root layout, Apple design tokens, dark mode
        page.tsx                  # Redirect to /new-run
        new-run/page.tsx          # Run launcher form
        live/[runId]/page.tsx     # Live execution + completed results
        history/page.tsx          # Run history table
        gallery/page.tsx          # Image gallery + detail panel
    components/
        Sidebar.tsx               # Persistent navigation sidebar
        StatsBar.tsx              # Live stats counters
        ProgressLog.tsx           # Streaming event log
        ImageGrid.tsx             # Thumbnail grid with verdict dots
        ImageDetail.tsx           # Score bars + audit trail
        ExportDropdown.tsx        # Download curated dataset (filter options)
    lib/
        api.ts                    # REST client
        ws.ts                     # WebSocket client
        types.ts                  # TypeScript types matching Pydantic schemas
    styles/
        globals.css               # Apple design system tokens + base styles
    tailwind.config.ts            # Apple color/font/spacing tokens
    package.json
    tsconfig.json

api/                              # FastAPI backend
    server.py                     # FastAPI app + CORS + startup
    routes/
        runs.py                   # POST/GET /api/runs, GET /api/runs/{id}
        images.py                 # GET /api/runs/{id}/images, GET /api/images
        export.py                 # GET /api/runs/{id}/export (ZIP)
        ws.py                     # WS /api/runs/{id}/stream
```

### Modified Existing Files
- `pyproject.toml` — Add fastapi, uvicorn, python-multipart to dependencies

---

## 6. Subsystem Decomposition

This is too large for a single implementation plan. Break into 3 sub-projects:

**Sub-project 1: FastAPI Backend**
- API server with all REST endpoints
- WebSocket bridge (EventBus → WS)
- Export endpoint (ZIP generation)
- Tests with mocked pipeline

**Sub-project 2: Next.js Shell + New Run + Live View**
- Next.js scaffold with Apple design system
- Sidebar, layout, dark mode
- New Run page (form → POST → redirect)
- Live View page (WebSocket → real-time UI)

**Sub-project 3: History + Gallery + Export UI**
- History page (REST → table)
- Gallery page (REST → grid + detail panel)
- Export dropdown component
- Bulk export from gallery

Build in this order: Backend first (frontend needs it), then Shell + Run + Live (the core flow), then History + Gallery (browsing).

---

## 7. What's Out of Scope

- User authentication (single-user research tool)
- Mobile responsive (desktop macOS target)
- Deployment/Docker (local development for now)
- Image upload to cloud storage (images stay local)
- Spec/plan approval flow in UI (auto-approve for now, can add later)

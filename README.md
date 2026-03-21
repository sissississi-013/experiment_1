# Validation Pipeline

An autonomous, multi-tier image dataset validation system that transforms natural language intent into executable validation programs. Built with LLM-driven orchestration, tiered tool execution, and enterprise-grade observability.

## What It Does

Given a natural language request like *"find 10 sharp, well-exposed images of horses from COCO"*, the pipeline autonomously:

1. **Resolves** the dataset (downloads from COCO, HuggingFace, or any URL)
2. **Generates** a formal validation specification from user intent via GPT-4o
3. **Calibrates** detection thresholds using Platt scaling on optional exemplar images
4. **Plans** a tiered validation strategy (cheap CPU checks first, expensive API calls last)
5. **Compiles** the plan into an executable program with early-exit optimization
6. **Executes** the program on every image, assigning verdicts (usable / recoverable / unusable)
7. **Supervises** results for anomalies (high flag rates, tool errors, degenerate thresholds)
8. **Reports** with per-image scores, curation metrics, and a full audit trail

Every step emits structured events for real-time observability. All runs persist to Neon Postgres for queryable history.

## Architecture

```
User Intent
    |
    v
[Module 0: Dataset Resolver]  -- GPT-4o resolves description -> COCO/HF/URL downloader
    |
    v
[Module 1: Spec Generator]    -- GPT-4o -> FormalSpec (quality + content criteria)
    |
    v
[Module 2: Calibrator]        -- Platt scaling on exemplars -> learned thresholds
    |
    v
[Module 3: Planner]           -- GPT-4o -> tiered ValidationPlan with tool_params
    |
    v
[Module 4: Compiler]          -- Plan -> CompiledProgram (executable lines + early exit)
    |
    v
[Module 5: Executor]          -- Per-image execution across 3 tiers
    |                            Tier 1 (CPU): blur, exposure, pixel stats
    |                            Tier 2 (API): NVIDIA GroundingDINO, Roboflow
    |                            Tier 3 (VLM): GPT-4o Vision semantic checks
    |
    v
[Module 6: Supervisor]        -- Anomaly detection (flag rates, error rates, empty results)
    |
    v
[Module 7: Reporter]          -- FinalReport with scores, audit trail, recommendations
    |
    v
[EventBus] --> CLI progress, Neon persistence, (future: WebSocket frontend)
```

## Tool Tiers

| Tier | Tools | Latency | Cost | What It Checks |
|------|-------|---------|------|----------------|
| 1 | Laplacian blur, histogram exposure, pixel stats | <5ms | Free | Image quality (sharpness, lighting, information content) |
| 2 | NVIDIA GroundingDINO, Roboflow | ~1-2s | Free tier | Content detection (does the image contain X?) |
| 3 | GPT-4o Vision | ~2-3s | API cost | Semantic quality (natural lighting? professional? cluttered?) |

Tier-based early exit: if an image fails Tier 1, Tier 2/3 are skipped. Most rejections happen at the cheapest tier.

## Tech Stack

- **Python 3.14** with Pydantic v2 for all data contracts
- **OpenAI GPT-4o** via `instructor` for structured LLM outputs (spec, plan, VLM checks)
- **NVIDIA NIM** for GroundingDINO object detection (serverless API)
- **OpenCV** for Tier 1 image quality analysis
- **Neon Postgres** for persistent run history (HTTP /sql endpoint)
- **scikit-learn** for Platt scaling (logistic regression calibration)

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Set up API keys in .env
OPENAI_API_KEY=<your-key>
NVIDIA_NIM_API_KEY=<your-key>
NEON_DATABASE_URL=<your-neon-connection-string>  # optional

# Run
python3 run_pipeline.py "find 10 sharp, well-exposed images of horses from COCO"
```

### CLI Options

```bash
# Auto-download from COCO
python3 run_pipeline.py "find 20 sharp images of dogs from COCO val2017"

# Use local dataset
python3 run_pipeline.py "find well-lit images" --dataset-path /path/to/images

# Explicit dataset description + intent
python3 run_pipeline.py "find sharp cat photos" --dataset-desc "50 cat images from COCO"
```

## Output

The CLI shows real-time progress as events stream in:

```
  [persistence] Connected to Neon (run_id=c8109182)
  [dataset_resolver] Started
  [dataset_resolver] Completed (1.8s)
  [dataset_resolver] Downloaded 10 images from coco
  [spec_generator] Completed (2.0s)
  [spec_generator] Spec: Identify 10 sharp, well-exposed horse images from COCO
  [planner] Plan: 3 steps across tiers [1, 2]
  [executor] 1/10 000000016228.jpg [~] recoverable (blur=1.00, exposure=0.52, content=0.41)
  [executor] 2/10 000000061171.jpg [O] usable (blur=1.00, exposure=0.53, content=0.67)
  ...
  [persistence] Run c8109182 saved to database
```

Final report includes dataset stats, curation score, per-image verdicts, and a full audit trail. Saved to `output/report.json`.

## Infrastructure

### Error Handling

Typed exception hierarchy (`PipelineError` -> `ToolError`, `LLMError`, `DatasetError`, `CalibrationError`, `SpecValidationError`) with per-image error collection. API tools use a shared `retry_with_policy` utility with exponential backoff. Errors are never swallowed — they're recorded per-image and surfaced in the report.

### Event System

A callback-based `EventBus` publishes typed Pydantic events (JSON-serializable) for every pipeline step. Subscribers include the CLI printer and Neon persistence. Designed for a future WebSocket/SSE frontend to plug in without changes.

### Persistence

Neon Postgres (serverless) stores all runs, events, and per-image results. Uses Neon's HTTP `/sql` endpoint (port 443) — works on any network. Optional: runs without a database if `NEON_DATABASE_URL` is not set.

```sql
-- Query across runs
SELECT image_id, verdict, scores->>'content' as content_score
FROM image_results WHERE verdict = 'usable' ORDER BY created_at DESC;
```

## Testing

```bash
# Unit tests (116 tests, ~2s)
python3 -m pytest -v -m "not integration"

# Integration test (requires API keys + network)
python3 -m pytest tests/test_end_to_end.py -v -m integration
```

## Project Structure

```
validation_pipeline/
    pipeline.py              # Orchestrator (chains 8 modules)
    config.py                # PipelineConfig + RetryPolicy
    errors.py                # Typed exception hierarchy
    events.py                # Pydantic event types
    event_bus.py             # Pub/sub EventBus
    retry.py                 # Exponential backoff retry utility
    modules/                 # 8 processing modules
    schemas/                 # 10 Pydantic data contracts
    tools/                   # Tool registry + 6 wrappers
        configs/             # YAML tool definitions
        wrappers/            # OpenCV, Roboflow, NVIDIA NIM, GPT-4o Vision
    dataset/                 # COCO, HuggingFace, URL downloaders
    persistence/             # Neon Postgres RunStore + EventBus subscriber
tests/                       # 116 tests across 22 files
```

# Progress & Event Streaming — Backend Polish Sprint A

**Date**: 2026-03-20
**Status**: Draft
**Sprint**: A (Event Streaming) — second of three backend polish sprints (B → A → C)

## Overview

Add a typed event system so every module publishes structured events as it runs. A simple callback-based EventBus routes events to subscribers. The CLI runner becomes the first subscriber (real-time progress output). The event system is designed so a future WebSocket/SSE frontend can plug in without changes.

## Why

Currently the pipeline runs silently — the CLI shows nothing until the final report. For a 10-image run that takes 30 seconds, the user stares at a blank screen. For Apple-grade UX, every step should be observable: what's running, what's in progress, what finished, what failed.

---

## 1. Event Types

**New file**: `validation_pipeline/events.py`

All events are Pydantic models — JSON-serializable out of the box for future WebSocket/SSE.

```python
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class PipelineEvent(BaseModel):
    """Base event. All events carry a timestamp and module name."""
    timestamp: datetime = Field(default_factory=datetime.now)
    module: str


# --- Lifecycle events ---

class ModuleStarted(PipelineEvent):
    """A module began processing."""
    details: str = ""


class ModuleCompleted(PipelineEvent):
    """A module finished processing."""
    duration_seconds: float = 0.0
    summary: str = ""


# --- Progress events ---

class ImageProgress(PipelineEvent):
    """Progress update during image processing."""
    current: int
    total: int
    image_path: str


class ToolProgress(PipelineEvent):
    """A single tool finished on a single image."""
    tool_name: str
    image_path: str
    score: float
    passed: bool


# --- Data events ---

class DatasetResolved(PipelineEvent):
    """Dataset resolver completed."""
    source: str
    image_count: int
    download_path: str


class SpecGenerated(PipelineEvent):
    """Spec generation completed."""
    spec_summary: str
    quality_criteria: list[str] = []
    content_criteria: list[str] = []


class PlanGenerated(PipelineEvent):
    """Plan generation completed."""
    steps_count: int
    tiers: list[int] = []


class ImageVerdict(PipelineEvent):
    """Verdict assigned to a single image."""
    image_id: str
    image_path: str
    verdict: str
    scores: dict[str, float] = {}
    errors: list[str] = []


# --- Error events ---

class PipelineErrorEvent(PipelineEvent):
    """An error occurred during processing."""
    error_type: str
    message: str
    context: dict[str, Any] = {}
```

---

## 2. EventBus

**New file**: `validation_pipeline/event_bus.py`

```python
import threading
from typing import Callable
from validation_pipeline.events import PipelineEvent


class EventBus:
    def __init__(self):
        self._subscribers: dict[type, list[Callable]] = {}
        self._all_subscribers: list[Callable] = []
        self._lock = threading.Lock()

    def subscribe(self, event_type: type[PipelineEvent], callback: Callable):
        """Register callback for a specific event type."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def subscribe_all(self, callback: Callable):
        """Register callback for ALL events."""
        with self._lock:
            self._all_subscribers.append(callback)

    def publish(self, event: PipelineEvent):
        """Publish event to matching subscribers. Never raises."""
        with self._lock:
            callbacks = list(self._subscribers.get(type(event), []))
            all_callbacks = list(self._all_subscribers)

        for cb in callbacks + all_callbacks:
            try:
                cb(event)
            except Exception:
                pass  # Subscriber errors never kill the pipeline

    def clear(self):
        """Remove all subscribers. For testing."""
        with self._lock:
            self._subscribers.clear()
            self._all_subscribers.clear()
```

Key design decisions:
- Thread-safe via lock (future-proofing)
- `publish` never raises — subscriber errors are swallowed so the pipeline is never affected
- `subscribe_all` is the hook for CLI printer and future frontend WebSocket bridge
- `clear()` for test isolation

---

## 3. Module Integration

**How modules get the bus:**

`ValidationPipeline.__init__` accepts an optional `event_bus: EventBus | None` parameter. If not provided, it creates one. Stored as `self.event_bus`.

Each module function gains an optional `event_bus: EventBus | None = None` parameter. If `None`, no events emitted. This keeps modules backwards-compatible and testable without a bus.

**Pipeline orchestrator** (`pipeline.py`) publishes lifecycle events around each module call:

```python
import time

bus = self.event_bus

# Module 0: Dataset Resolution
bus.publish(ModuleStarted(module="dataset_resolver"))
t = time.time()
# ... resolve and download ...
# image_count derived from listing the download_path directory after download
bus.publish(DatasetResolved(module="dataset_resolver", source=plan.source, image_count=len(images), download_path=local_path))
bus.publish(ModuleCompleted(module="dataset_resolver", duration_seconds=time.time()-t))

# Module 1: Spec Generator
bus.publish(ModuleStarted(module="spec_generator"))
t = time.time()
spec = generate_spec(user_input, self.config)  # no event_bus param needed
bus.publish(SpecGenerated(module="spec_generator", spec_summary=spec.restated_request, ...))
bus.publish(ModuleCompleted(module="spec_generator", duration_seconds=time.time()-t))
```

Note: Module functions (spec_generator, planner, calibrator, compiler, supervisor, reporter) do NOT receive `event_bus`. Only `execute_program` receives it, because it publishes per-image progress from inside its loop.

**Executor** publishes per-image progress and verdicts from `execute_program`'s main loop (NOT from `_run_program_on_image`):

```python
# execute_program signature gains event_bus:
def execute_program(program, dataset_path, tools, calibration=None, sample_paths=None, event_bus=None):

# Inside the image loop in execute_program:
for i, img_path in enumerate(image_paths):
    if event_bus:
        event_bus.publish(ImageProgress(module="executor", current=i+1, total=len(image_paths), image_path=img_path))
    # ... process image ...
    if event_bus:
        scores = {tr.dimension: tr.score for tr in img_result.tool_results}
        event_bus.publish(ImageVerdict(module="executor", image_id=img_result.image_id, image_path=img_path, verdict=img_result.verdict, scores=scores, errors=img_result.errors))
```

`ToolProgress` events are published from inside `_run_program_on_image` (which receives `event_bus` passed down from `execute_program`):

```python
# After each tool completes in _run_program_on_image:
if event_bus:
    event_bus.publish(ToolProgress(module="executor", tool_name=tool_name, image_path=image_path, score=tr.score, passed=passed))
```

**Error events** published from the pipeline orchestrator. Each module call is wrapped in try/except:

```python
# Around each module call in pipeline.run():
try:
    spec = generate_spec(user_input, self.config)
except PipelineError as e:
    bus.publish(PipelineErrorEvent(module=e.module, error_type=type(e).__name__, message=str(e), context=e.context))
    raise
```

### Event publishing locations:
- **`pipeline.py`** — All lifecycle events (ModuleStarted/Completed), data events (SpecGenerated, PlanGenerated, DatasetResolved), error events (PipelineErrorEvent). This is the central coordination point.
- **`executor.py`** (`execute_program` + `_run_program_on_image`) — ImageProgress, ToolProgress, ImageVerdict. These are the only events published from inside a module, because the executor has per-image iteration that pipeline.py doesn't see.

All other modules (spec_generator, planner, calibrator, compiler, supervisor, reporter) do NOT receive `event_bus` and do NOT publish events. Pipeline.py handles their lifecycle.

This keeps event publishing concentrated in two files only.

---

## 4. CLI Consumer

**Modify**: `run_pipeline.py`

Replace the current "run and then print report" approach with a real-time subscriber that prints progress as events arrive.

**Full wiring in `run_pipeline.py`:**

```python
from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import *

def cli_subscriber(event: PipelineEvent):
    if isinstance(event, ModuleStarted):
        print(f"[{event.module}] Started {event.details}")
    elif isinstance(event, ModuleCompleted):
        print(f"[{event.module}] Completed ({event.duration_seconds:.1f}s) {event.summary}")
    elif isinstance(event, ImageProgress):
        print(f"[executor] {event.current}/{event.total} {os.path.basename(event.image_path)}", end="")
    elif isinstance(event, ImageVerdict):
        scores = ", ".join(f"{k}={v:.2f}" for k, v in event.scores.items())
        print(f" → {event.verdict} ({scores})")
    elif isinstance(event, PipelineErrorEvent):
        print(f"[{event.module}] ERROR: {event.message}")

def run(...):
    bus = EventBus()
    bus.subscribe_all(cli_subscriber)

    config = PipelineConfig(openai_api_key=...)
    pipeline = ValidationPipeline(config, event_bus=bus)
    report = pipeline.run(user_input, auto_approve=True)

    # Full report still printed at the end
    print_report(report)
```

The user sees every step as it happens, then the full report at the end.

---

## 5. Test Strategy

- `tests/test_events.py` — Event type construction, JSON serialization
- `tests/test_event_bus.py` — subscribe, publish, subscribe_all, subscriber error isolation, clear, thread safety
- Updated pipeline test — verify events are published during a run (subscribe and collect)

---

## 6. File Structure

### New Files
```
validation_pipeline/events.py        # Event type hierarchy
validation_pipeline/event_bus.py     # EventBus pub/sub
tests/test_events.py                 # Event type tests
tests/test_event_bus.py              # EventBus tests
```

### Modified Files
```
validation_pipeline/pipeline.py      # Create EventBus, publish lifecycle/data events
validation_pipeline/modules/executor.py   # Publish ImageProgress, ImageVerdict
run_pipeline.py                      # Subscribe to events, print real-time progress
```

---

## 7. What's Explicitly Out of Scope

- No WebSocket/SSE server (that's frontend work)
- No structured log file output (could be added as another subscriber later)
- No async EventBus (synchronous is fine for now)
- No changes to schemas, tools, or report structure
- Modules other than executor don't publish their own events — pipeline.py handles lifecycle

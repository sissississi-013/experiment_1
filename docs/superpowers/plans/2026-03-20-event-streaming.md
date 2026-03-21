# Event Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a typed event system so every pipeline step emits structured events, with a CLI subscriber for real-time progress output.

**Architecture:** Pydantic-based event types published to a callback-based EventBus. The pipeline orchestrator publishes lifecycle events around each module call. The executor publishes per-image progress. `run_pipeline.py` subscribes and prints real-time progress.

**Tech Stack:** Python 3.14, Pydantic, threading (for lock)

**Spec:** `docs/superpowers/specs/2026-03-20-event-streaming-design.md`

---

### Task 1: Event types

**Files:**
- Create: `validation_pipeline/events.py`
- Test: `tests/test_events.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_events.py
import json
from datetime import datetime
from validation_pipeline.events import (
    PipelineEvent, ModuleStarted, ModuleCompleted,
    ImageProgress, ToolProgress, ImageVerdict,
    SpecGenerated, PlanGenerated, DatasetResolved,
    PipelineErrorEvent,
)


def test_pipeline_event_has_timestamp_and_module():
    event = PipelineEvent(module="test")
    assert event.module == "test"
    assert isinstance(event.timestamp, datetime)


def test_module_started_event():
    event = ModuleStarted(module="executor", details="processing 10 images")
    assert event.module == "executor"
    assert event.details == "processing 10 images"


def test_image_progress_event():
    event = ImageProgress(module="executor", current=3, total=10, image_path="/img/003.jpg")
    assert event.current == 3
    assert event.total == 10


def test_image_verdict_event():
    event = ImageVerdict(
        module="executor", image_id="003", image_path="/img/003.jpg",
        verdict="usable", scores={"blur": 0.95, "exposure": 0.6},
    )
    assert event.verdict == "usable"
    assert event.scores["blur"] == 0.95


def test_events_are_json_serializable():
    event = ModuleCompleted(module="spec_generator", duration_seconds=2.1, summary="done")
    data = json.loads(event.model_dump_json())
    assert data["module"] == "spec_generator"
    assert data["duration_seconds"] == 2.1


def test_pipeline_error_event():
    event = PipelineErrorEvent(
        module="executor", error_type="ToolError",
        message="API timeout", context={"tool": "nvidia"},
    )
    assert event.error_type == "ToolError"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_events.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement events.py**

```python
# validation_pipeline/events.py
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class PipelineEvent(BaseModel):
    """Base event. All events carry a timestamp and module name."""
    timestamp: datetime = Field(default_factory=datetime.now)
    module: str


class ModuleStarted(PipelineEvent):
    """A module began processing."""
    details: str = ""


class ModuleCompleted(PipelineEvent):
    """A module finished processing."""
    duration_seconds: float = 0.0
    summary: str = ""


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


class PipelineErrorEvent(PipelineEvent):
    """An error occurred during processing."""
    error_type: str
    message: str
    context: dict[str, Any] = {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_events.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/events.py tests/test_events.py
git commit -m "feat: add typed pipeline event hierarchy"
```

---

### Task 2: EventBus

**Files:**
- Create: `validation_pipeline/event_bus.py`
- Test: `tests/test_event_bus.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_event_bus.py
from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import PipelineEvent, ModuleStarted, ModuleCompleted, ImageProgress


def test_subscribe_and_publish():
    bus = EventBus()
    received = []
    bus.subscribe(ModuleStarted, lambda e: received.append(e))
    bus.publish(ModuleStarted(module="test"))
    assert len(received) == 1
    assert received[0].module == "test"


def test_subscribe_does_not_receive_other_types():
    bus = EventBus()
    received = []
    bus.subscribe(ModuleStarted, lambda e: received.append(e))
    bus.publish(ModuleCompleted(module="test"))
    assert len(received) == 0


def test_subscribe_all_receives_everything():
    bus = EventBus()
    received = []
    bus.subscribe_all(lambda e: received.append(e))
    bus.publish(ModuleStarted(module="a"))
    bus.publish(ModuleCompleted(module="b"))
    bus.publish(ImageProgress(module="c", current=1, total=10, image_path="/x.jpg"))
    assert len(received) == 3


def test_subscriber_error_does_not_kill_pipeline():
    bus = EventBus()
    received = []

    def bad_subscriber(event):
        raise RuntimeError("subscriber crash")

    def good_subscriber(event):
        received.append(event)

    bus.subscribe_all(bad_subscriber)
    bus.subscribe_all(good_subscriber)
    bus.publish(ModuleStarted(module="test"))
    assert len(received) == 1  # good subscriber still got the event


def test_clear_removes_all_subscribers():
    bus = EventBus()
    received = []
    bus.subscribe_all(lambda e: received.append(e))
    bus.clear()
    bus.publish(ModuleStarted(module="test"))
    assert len(received) == 0


def test_multiple_subscribers_same_type():
    bus = EventBus()
    a, b = [], []
    bus.subscribe(ModuleStarted, lambda e: a.append(e))
    bus.subscribe(ModuleStarted, lambda e: b.append(e))
    bus.publish(ModuleStarted(module="test"))
    assert len(a) == 1
    assert len(b) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_event_bus.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement event_bus.py**

```python
# validation_pipeline/event_bus.py
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
                pass

    def clear(self):
        """Remove all subscribers. For testing."""
        with self._lock:
            self._subscribers.clear()
            self._all_subscribers.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_event_bus.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/event_bus.py tests/test_event_bus.py
git commit -m "feat: add callback-based EventBus with type routing"
```

---

### Task 3: Pipeline orchestrator — publish lifecycle events

**Files:**
- Modify: `validation_pipeline/pipeline.py:17-108`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_pipeline.py
from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import ModuleStarted, ModuleCompleted, SpecGenerated, PlanGenerated


def test_pipeline_publishes_lifecycle_events(tmp_path):
    """Pipeline emits ModuleStarted/Completed events for each step."""
    img_dir = tmp_path / "dataset"
    img_dir.mkdir()
    for i in range(5):
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_dir / f"img_{i:03d}.jpg"))

    config_dir = tmp_path / "tool_configs"
    config_dir.mkdir()
    (config_dir / "laplacian_blur.yaml").write_text(
        "name: laplacian_blur\ntask_type: image_quality\ntier: 1\nsource: local\n"
        'wrapper_class: "opencv_wrapper.LaplacianBlurTool"\ndefault_config: {}\ncost_estimate_ms: 1\n'
    )

    mock_spec = FormalSpec(
        restated_request="Test", assumptions=[], content_criteria=[],
        quality_criteria=[QualityCriterion(dimension="blur", description="sharp")],
        quantity_targets=QuantityTarget(), output_format=OutputFormat(),
        success_criteria="test", user_confirmed=True,
    )
    mock_plan = ValidationPlan(
        plan_id="p1", spec_summary="test", sampling_strategy=SamplingStrategy(),
        steps=[PlanStep(step_id=1, dimension="blur", tool_name="laplacian_blur",
            threshold=100.0, threshold_source="default", hypothesis="test", tier=1)],
        combination_logic="ALL_PASS", estimated_cost=CostEstimate(), user_approved=True,
    )

    events_received = []
    bus = EventBus()
    bus.subscribe_all(lambda e: events_received.append(e))

    config = PipelineConfig(tool_configs_dir=str(config_dir))
    pipeline = ValidationPipeline(config, event_bus=bus)

    with patch("validation_pipeline.modules.spec_generator._call_llm", return_value=mock_spec), \
         patch("validation_pipeline.modules.planner._call_llm", return_value=mock_plan):
        pipeline.run(UserInput(dataset_path=str(img_dir), intent="test"), auto_approve=True)

    module_names = [e.module for e in events_received if isinstance(e, ModuleStarted)]
    assert "spec_generator" in module_names
    assert "executor" in module_names
    assert "reporter" in module_names

    completed_names = [e.module for e in events_received if isinstance(e, ModuleCompleted)]
    assert "spec_generator" in completed_names
    assert "executor" in completed_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_pipeline.py::test_pipeline_publishes_lifecycle_events -v`
Expected: FAIL — ValidationPipeline doesn't accept event_bus

- [ ] **Step 3: Update pipeline.py**

Add imports at top:
```python
import time
from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import (
    ModuleStarted, ModuleCompleted, SpecGenerated, PlanGenerated,
    DatasetResolved, PipelineErrorEvent,
)
from validation_pipeline.errors import PipelineError
```

Update `__init__`:
```python
    def __init__(self, config: PipelineConfig | None = None, event_bus: EventBus | None = None):
        self.config = config or PipelineConfig()
        self.event_bus = event_bus or EventBus()
        self.registry = ToolRegistry(self.config.tool_configs_dir)
```

Wrap each module call in the `run()` method with lifecycle events and error catching. The pattern for each module:
```python
        # Module N: <Name>
        self.event_bus.publish(ModuleStarted(module="<name>"))
        t = time.time()
        try:
            # ... existing module call ...
        except PipelineError as e:
            self.event_bus.publish(PipelineErrorEvent(
                module=e.module, error_type=type(e).__name__,
                message=str(e), context=e.context,
            ))
            raise
        self.event_bus.publish(ModuleCompleted(module="<name>", duration_seconds=time.time()-t))
```

After spec generation, publish:
```python
        self.event_bus.publish(SpecGenerated(
            module="spec_generator", spec_summary=spec.restated_request,
            quality_criteria=[qc.dimension for qc in spec.quality_criteria],
            content_criteria=[cc.object_or_scene for cc in spec.content_criteria],
        ))
```

After plan generation, publish:
```python
        self.event_bus.publish(PlanGenerated(
            module="planner", steps_count=len(plan.steps),
            tiers=sorted(set(s.tier for s in plan.steps)),
        ))
```

After dataset resolution, publish:
```python
        self.event_bus.publish(DatasetResolved(
            module="dataset_resolver", source=dataset_plan.source,
            image_count=len(list(Path(local_path).iterdir())),
            download_path=local_path,
        ))
```

Pass `event_bus` to executor:
```python
        result = execute_program(program, user_input.dataset_path, tools, cal_result, event_bus=self.event_bus)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add validation_pipeline/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline publishes lifecycle events via EventBus"
```

---

### Task 4: Executor — publish per-image progress events

**Files:**
- Modify: `validation_pipeline/modules/executor.py:13-19,34-48,62-67,91-98`
- Test: `tests/test_executor.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_executor.py
from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import ImageProgress, ImageVerdict, ToolProgress


def test_executor_publishes_progress_events(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_dir / f"img_{i}.jpg"))

    class SimpleTool(BaseTool):
        name = "simple_tool"
        task_type = "image_quality"
        tier = 1
        def execute(self, image, **kwargs):
            return 0.8
        def normalize(self, raw_output, calibration=None):
            return ToolResult(
                tool_name="simple_tool", dimension="blur",
                score=0.8, passed=True, threshold=0.5, raw_output=0.8,
            )

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[ProgramLine(
            line_number=1, variable_name="blur_score",
            tool_call="simple_tool(image)", output_type="float",
            threshold_check="blur_score >= 0.5", tier=1,
        )],
        batch_strategy=BatchStrategy(),
        tool_imports=["simple_tool"],
    )

    events = []
    bus = EventBus()
    bus.subscribe_all(lambda e: events.append(e))

    tools = {"simple_tool": SimpleTool({})}
    result = execute_program(program, str(img_dir), tools, event_bus=bus)

    progress_events = [e for e in events if isinstance(e, ImageProgress)]
    assert len(progress_events) == 3
    assert progress_events[0].current == 1
    assert progress_events[2].current == 3

    verdict_events = [e for e in events if isinstance(e, ImageVerdict)]
    assert len(verdict_events) == 3
    assert all(v.verdict == "usable" for v in verdict_events)

    tool_events = [e for e in events if isinstance(e, ToolProgress)]
    assert len(tool_events) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_executor.py::test_executor_publishes_progress_events -v`
Expected: FAIL — execute_program doesn't accept event_bus

- [ ] **Step 3: Update executor**

Add import at top of `validation_pipeline/modules/executor.py`:
```python
from validation_pipeline.events import ImageProgress, ImageVerdict, ToolProgress
```

Add `event_bus=None` parameter to `execute_program` (after `sample_paths`):
```python
def execute_program(
    program: CompiledProgram,
    dataset_path: str,
    tools: dict[str, BaseTool],
    calibration: CalibrationResult | None = None,
    sample_paths: list[str] | None = None,
    event_bus=None,
) -> ExecutionResult:
```

In the image loop (line 34), add progress event and pass event_bus to `_run_program_on_image`:
```python
    for i, img_path in enumerate(image_paths):
        if event_bus:
            event_bus.publish(ImageProgress(module="executor", current=i+1, total=len(image_paths), image_path=img_path))
        try:
            img = Image.open(img_path).convert("RGB")
            img_result = _run_program_on_image(program, img, img_path, tools, calibration, event_bus=event_bus)
            results.append(img_result)
            if event_bus:
                scores = {tr.dimension: tr.score for tr in img_result.tool_results}
                event_bus.publish(ImageVerdict(module="executor", image_id=img_result.image_id, image_path=img_path, verdict=img_result.verdict, scores=scores, errors=img_result.errors))
        except Exception as e:
            ...
```

Add `event_bus=None` to `_run_program_on_image` signature:
```python
def _run_program_on_image(program, image, image_path, tools, calibration, event_bus=None):
```

After each tool completes (after `tr.passed = passed`, around line 116), publish ToolProgress:
```python
        if event_bus:
            event_bus.publish(ToolProgress(module="executor", tool_name=tool_name, image_path=image_path, score=tr.score, passed=passed))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_executor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add validation_pipeline/modules/executor.py tests/test_executor.py
git commit -m "feat: executor publishes ImageProgress, ImageVerdict, ToolProgress events"
```

---

### Task 5: CLI consumer — real-time progress in run_pipeline.py

**Files:**
- Modify: `run_pipeline.py`

- [ ] **Step 1: Update run_pipeline.py**

Add imports at the top (after existing imports):
```python
from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import (
    PipelineEvent, ModuleStarted, ModuleCompleted, ImageProgress,
    ImageVerdict, SpecGenerated, PlanGenerated, DatasetResolved,
    PipelineErrorEvent, ToolProgress,
)
```

Add the CLI subscriber function (before the `run()` function):
```python
def cli_subscriber(event: PipelineEvent):
    """Print real-time progress to console."""
    if isinstance(event, ModuleStarted):
        detail = f" — {event.details}" if event.details else ""
        print(f"  [{event.module}] Started{detail}")
    elif isinstance(event, ModuleCompleted):
        summary = f" — {event.summary}" if event.summary else ""
        print(f"  [{event.module}] Completed ({event.duration_seconds:.1f}s){summary}")
    elif isinstance(event, DatasetResolved):
        print(f"  [dataset_resolver] Downloaded {event.image_count} images from {event.source}")
    elif isinstance(event, SpecGenerated):
        print(f"  [spec_generator] Spec: {event.spec_summary}")
    elif isinstance(event, PlanGenerated):
        print(f"  [planner] Plan: {event.steps_count} steps across tiers {event.tiers}")
    elif isinstance(event, ImageProgress):
        print(f"  [executor] {event.current}/{event.total} {os.path.basename(event.image_path)}", end="", flush=True)
    elif isinstance(event, ImageVerdict):
        scores = ", ".join(f"{k}={v:.2f}" for k, v in event.scores.items())
        icon = {"usable": "O", "recoverable": "~", "unusable": "X", "error": "!"}
        print(f" [{icon.get(event.verdict, '?')}] {event.verdict} ({scores})")
    elif isinstance(event, PipelineErrorEvent):
        print(f"  [{event.module}] ERROR: {event.message}")
```

Update the `run()` function to create EventBus and pass to pipeline:
```python
def run(intent, dataset_path=None, dataset_description=None):
    bus = EventBus()
    bus.subscribe_all(cli_subscriber)

    config = PipelineConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    pipeline = ValidationPipeline(config, event_bus=bus)

    # ... rest of run() stays the same ...
```

Remove the `print_section("Running pipeline (auto-approve mode)")` line since events now show this.

- [ ] **Step 2: Run full test suite to verify no regressions**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add run_pipeline.py
git commit -m "feat: CLI runner shows real-time progress via event subscription"
```

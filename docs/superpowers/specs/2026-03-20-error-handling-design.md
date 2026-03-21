# Error Handling & Resilience — Backend Polish Sprint B

**Date**: 2026-03-20
**Status**: Draft
**Sprint**: B (Error Handling) — first of three backend polish sprints (B → A → C)

## Overview

Replace silent error swallowing throughout the pipeline with a typed exception hierarchy, per-image error collection, and a shared retry policy. Every failure becomes visible, traceable, and actionable.

## Why

Currently, when a tool API times out or returns garbage, the pipeline silently returns `score=0.0` and marks the image as "unusable" with no explanation. This is misleading — the image might be perfectly fine, the API just failed. For Apple-grade engineering, every failure must be explicit: what failed, where, why, and whether it was retried.

---

## 1. Exception Hierarchy

**New file**: `validation_pipeline/errors.py`

```python
class PipelineError(Exception):
    """Base for all pipeline errors."""
    def __init__(self, message: str, module: str, context: dict | None = None):
        self.module = module
        self.context = context or {}
        super().__init__(message)

class DatasetError(PipelineError):
    """Download failures, missing paths, bad formats, category not found."""
    pass

class LLMError(PipelineError):
    """OpenAI/instructor failures, invalid structured responses, retry exhaustion."""
    pass

class ToolError(PipelineError):
    """Tool execution failures — API timeout, bad image, rate limiting."""
    pass

class CalibrationError(PipelineError):
    """Not enough exemplars, degenerate data, failed Platt fitting."""
    pass

class SpecValidationError(PipelineError):
    """Spec/plan validation failures — missing fields, invalid values."""
    pass
```

Every exception carries:
- `message`: Human-readable description
- `module`: Which module raised it (e.g., `"executor"`, `"dataset_resolver"`)
- `context`: Dict with relevant details (image path, tool name, HTTP status, retry count, etc.)

---

## 2. Shared Retry Policy

**New file**: `validation_pipeline/retry.py`

### RetryPolicy config

Added to `validation_pipeline/config.py`:

```python
class RetryPolicy(BaseModel):
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
```

Added to `PipelineConfig`:
```python
class PipelineConfig(BaseModel):
    # ... existing fields ...
    retry_policy: RetryPolicy = RetryPolicy()
```

### retry_with_policy utility

```python
def retry_with_policy(
    fn: Callable[[], T],
    policy: RetryPolicy,
    error_cls: type[PipelineError],
    module: str,
    context: dict | None = None,
) -> T:
    """
    Call fn() with retries according to policy.
    On transient failures (timeouts, 5xx, rate limits): retry with exponential backoff.
    On permanent failures (4xx, auth, validation): raise immediately.
    On retry exhaustion: raise error_cls with full context including retry count.
    Respects Retry-After headers when available.
    """
```

This replaces the duplicated `for attempt in range(self.max_retries)` loops in tool wrappers. The `fn` parameter is a zero-arg callable — callers use closures/lambdas to capture image data and params (e.g., `lambda: self._run_inference(b64_image, target_label)`).

**Transient vs permanent errors:**
- Transient (retry): `TimeoutError`, `ConnectionError`, HTTP 429, HTTP 5xx
- Permanent (raise immediately): HTTP 401/403 (auth), HTTP 400 (bad request)

**Interaction with `instructor` retries:** The LLM-calling modules (spec_generator, planner, dataset_resolver) already use `instructor`'s built-in `max_retries` for Pydantic validation errors (malformed structured output). `retry_with_policy` is NOT applied to these — `instructor` handles its own retry logic. Instead, the LLM modules wrap the `instructor` call in a try/except and raise `LLMError` on final failure. `retry_with_policy` is only for the tool wrappers (Roboflow, NIM, GPT-4o Vision) which do raw HTTP calls.

**Deprecating `PipelineConfig.max_retries`:** The existing `max_retries: int = 3` on `PipelineConfig` continues to be used for `instructor` calls. `RetryPolicy` governs tool wrapper retries. These are separate concerns.

---

## 3. Per-Module Error Handling Changes

### 3.1 ImageResult schema change

**Modify**: `validation_pipeline/schemas/execution.py`

```python
class ImageResult(BaseModel):
    image_id: str
    image_path: str
    tool_results: list[ToolResult]
    verdict: str              # Now includes "error" as a valid verdict
    verdict_reason: str
    errors: list[str] = []    # NEW: collected error messages for this image
    lines_executed: int = 0
    exemplar_similarity: float | None = None
```

New verdict value `"error"` — means the image couldn't be properly evaluated due to tool/API failures (distinct from "unusable" which means the image was evaluated and failed).

### 3.1.1 ExecutionSummary schema change

**Modify**: `validation_pipeline/schemas/execution.py`

Add `error_count: int = 0` to `ExecutionSummary`. Update `_compute_summary()` in `executor.py` to count `verdict="error"` images and populate `tool_error_rate` from per-image error data.

### 3.2 Executor

**Modify**: `validation_pipeline/modules/executor.py`

Current behavior: catches all exceptions per-image, returns `verdict="unusable"` with `"Processing error: {e}"`.

New behavior:
- Tool execution failure → catch `ToolError`, record in `image_result.errors`, skip that tool's result
- If ALL tools errored for an image → `verdict="error"`, `verdict_reason="All tools failed: {error_list}"`
- If SOME tools errored but others ran → use available results for verdict, record errors
- Tool returning 0.0 is a real score; tool raising `ToolError` is a failure — these are now distinct
- Uses `retry_with_policy` instead of inline retry loops

### 3.3 Dataset Resolver

**Modify**: `validation_pipeline/modules/dataset_resolver.py`

- `_call_llm` failure → wrap `instructor` call in try/except, raise `LLMError(module="dataset_resolver", context={"description": ...})` (instructor handles its own retries internally)
- Download failure → `DatasetError(module="dataset_resolver", context={"url": ..., "http_status": ..., "source": ...})`
- Category not found → `DatasetError(module="dataset_resolver", context={"category": ..., "available": [...]})`
- Unknown dataset source → `DatasetError(module="dataset_resolver", context={"source": ..., "supported": [...]})`
- Missing annotation file → `DatasetError(module="dataset_resolver", context={"subset": ..., "cache_dir": ...})`

### 3.4 Spec Generator

**Modify**: `validation_pipeline/modules/spec_generator.py`

- `_call_llm` failure → wrap `instructor` call in try/except, raise `LLMError(module="spec_generator", context={"intent": ...})` (instructor handles its own retries internally)

### 3.5 Planner

**Modify**: `validation_pipeline/modules/planner.py`

- `_call_llm` failure → wrap `instructor` call in try/except, raise `LLMError(module="planner", context={"spec_summary": ...})` (instructor handles its own retries internally)

### 3.6 Calibrator

**Modify**: `validation_pipeline/modules/calibrator.py`

- No exemplars provided (empty lists) → skip calibration gracefully, return empty `CalibrationResult` with a clear `threshold_report` explaining why (no longer produces numpy warnings)
- Degenerate data (all same score, <2 classes) → already handled, but now wraps in `CalibrationError` if the result is completely unusable
- Tool execution failure on a single image during calibration → skip that image, collect the error, continue with remaining images. If ALL images fail → raise `CalibrationError(module="calibrator", context={"dimension": ..., "failed_count": ..., "total_count": ...})`

### 3.7 Tool Wrappers

**Modify**: All three API tool wrappers (NIM, Roboflow, GPT-4o Vision)

Current behavior: Each has its own `for attempt in range(self.max_retries)` loop. On final failure, silently returns `{"best_confidence": 0.0}`.

New behavior:
- Remove inline retry loops
- `execute()` calls the API via `retry_with_policy`
- On retry exhaustion: raises `ToolError(module="nvidia_grounding_dino", context={"target_label": ..., "http_status": ..., "retry_count": 3})`
- The executor catches this `ToolError` and records it per-image
- No more silent 0.0 scores on API failure

Tier 1 tools (OpenCV) don't change — they don't have retry logic and failures are genuine (bad image data). If a Tier 1 tool raises an unexpected exception (e.g., corrupt image causing numpy error), the executor catches it as a generic `Exception`, wraps it in `ToolError`, and records it per-image.

### 3.8 Compiler

**Modify**: `validation_pipeline/modules/compiler.py`

- Currently raises `ValueError` if plan is not approved. Change to `SpecValidationError(module="compiler", context={"plan_id": ...})`
- Malformed step data → `SpecValidationError` with step details

### 3.9 Supervisor

**Modify**: `validation_pipeline/modules/supervisor.py`

- Now receives `error_count` from `ExecutionSummary` and factors it into anomaly detection
- High error rate (>10% of images with `verdict="error"`) → blocker anomaly with `"likely_cause": "Tool API failures"`
- Populates `tool_error_rate` from per-image error data passed through from executor

### 3.10 Reporter

**Modify**: `validation_pipeline/modules/reporter.py`

- Includes error images in `per_image_results` with `verdict="error"` and their `errors` list
- `DatasetStats` now tracks error count alongside usable/recoverable/unusable

### 3.11 Pipeline Orchestrator

**Modify**: `validation_pipeline/pipeline.py`

- Each module call wrapped in try/except for the appropriate `PipelineError` subtype
- Dataset resolution failure → re-raise `DatasetError` (caller decides what to do)
- Spec/plan generation failure → re-raise `LLMError`
- Execution proceeds even if some images error — the report shows which ones
- Supervision and reporting receive the error information and include it in the final report

---

## 4. Test Strategy

Each new component gets focused tests:

- `tests/test_errors.py` — Exception hierarchy, context propagation, string representation
- `tests/test_retry.py` — Retry policy: transient retries, permanent immediate raise, backoff timing, exhaustion
- Updated executor tests — New `"error"` verdict, error collection, partial tool failures
- Updated tool wrapper tests — Verify `ToolError` raised (not silent 0.0) on retry exhaustion
- Updated module tests — Verify correct exception types with correct context

---

## 5. File Structure

### New Files
```
validation_pipeline/errors.py          # Exception hierarchy
validation_pipeline/retry.py           # retry_with_policy utility
tests/test_errors.py                   # Exception tests
tests/test_retry.py                    # Retry policy tests
```

### Modified Files
```
validation_pipeline/config.py                        # Add RetryPolicy
validation_pipeline/schemas/execution.py             # Add errors field to ImageResult
validation_pipeline/modules/executor.py              # Error collection, "error" verdict
validation_pipeline/modules/dataset_resolver.py      # Typed exceptions
validation_pipeline/modules/spec_generator.py        # Typed exceptions
validation_pipeline/modules/planner.py               # Typed exceptions
validation_pipeline/modules/calibrator.py            # Typed exceptions, clean no-exemplar handling
validation_pipeline/tools/wrappers/nvidia_nim_wrapper.py    # Use retry_with_policy, raise ToolError
validation_pipeline/tools/wrappers/roboflow_wrapper.py      # Use retry_with_policy, raise ToolError
validation_pipeline/tools/wrappers/openai_vision_wrapper.py # Use retry_with_policy, raise ToolError
validation_pipeline/modules/compiler.py               # SpecValidationError
validation_pipeline/modules/supervisor.py             # Error rate anomaly detection
validation_pipeline/modules/reporter.py               # Error images in report
validation_pipeline/pipeline.py                       # Per-step error handling
```

---

## 6. What's Explicitly Out of Scope

- No logging system (comes in Sprint A: Progress & Event Streaming)
- No database persistence (comes in Sprint C: Persistence)
- No new CLI flags or UI changes
- No changes to tool registry or YAML configs
- Tier 1 OpenCV tool code unchanged (no API, no retry needed — but executor now wraps their exceptions too)

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

class ValidationError(PipelineError):
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

This replaces the duplicated `for attempt in range(self.max_retries)` loops in every tool wrapper and LLM caller.

**Transient vs permanent errors:**
- Transient (retry): `TimeoutError`, `ConnectionError`, HTTP 429, HTTP 5xx
- Permanent (raise immediately): HTTP 401/403 (auth), HTTP 400 (bad request), `instructor` validation errors

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

- `_call_llm` failure → `LLMError(module="dataset_resolver", context={"description": ..., "retry_count": ...})`
- Download failure → `DatasetError(module="dataset_resolver", context={"url": ..., "http_status": ..., "source": ...})`
- Category not found → `DatasetError(module="dataset_resolver", context={"category": ..., "available": [...]})`
- Uses `retry_with_policy` for the LLM call

### 3.4 Spec Generator

**Modify**: `validation_pipeline/modules/spec_generator.py`

- `_call_llm` failure → `LLMError(module="spec_generator", context={"intent": ..., "retry_count": ...})`
- Uses `retry_with_policy` for the LLM call

### 3.5 Planner

**Modify**: `validation_pipeline/modules/planner.py`

- `_call_llm` failure → `LLMError(module="planner", context={"spec_summary": ..., "retry_count": ...})`
- Uses `retry_with_policy` for the LLM call

### 3.6 Calibrator

**Modify**: `validation_pipeline/modules/calibrator.py`

- No exemplars provided (empty lists) → skip calibration gracefully, return empty `CalibrationResult` with a clear `threshold_report` explaining why (no longer produces numpy warnings)
- Degenerate data (all same score, <2 classes) → already handled, but now wraps in `CalibrationError` if the result is completely unusable
- Tool execution failure during calibration → `ToolError` with image path and tool name

### 3.7 Tool Wrappers

**Modify**: All three API tool wrappers (NIM, Roboflow, GPT-4o Vision)

Current behavior: Each has its own `for attempt in range(self.max_retries)` loop. On final failure, silently returns `{"best_confidence": 0.0}`.

New behavior:
- Remove inline retry loops
- `execute()` calls the API via `retry_with_policy`
- On retry exhaustion: raises `ToolError(module="nvidia_grounding_dino", context={"target_label": ..., "http_status": ..., "retry_count": 3})`
- The executor catches this `ToolError` and records it per-image
- No more silent 0.0 scores on API failure

Tier 1 tools (OpenCV) don't change — they don't have retry logic and failures are genuine (bad image data).

### 3.8 Pipeline Orchestrator

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
validation_pipeline/pipeline.py                      # Per-step error handling
```

---

## 6. What's Explicitly Out of Scope

- No logging system (comes in Sprint A: Progress & Event Streaming)
- No database persistence (comes in Sprint C: Persistence)
- No new CLI flags or UI changes
- No changes to tool registry, YAML configs, or report schemas (beyond ImageResult.errors)
- Tier 1 OpenCV tools unchanged (no API, no retry needed)

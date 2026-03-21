# Error Handling & Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace silent error swallowing with typed exceptions, shared retry policy, and per-image error collection across the entire pipeline.

**Architecture:** A custom exception hierarchy (`PipelineError` → typed subclasses) provides structured error context. A shared `retry_with_policy` utility handles transient failures for API tool wrappers. The executor collects errors per-image instead of silently scoring 0.0. LLM modules wrap `instructor` calls and raise `LLMError` on failure.

**Tech Stack:** Python 3.14, Pydantic, requests, instructor/OpenAI

**Spec:** `docs/superpowers/specs/2026-03-20-error-handling-design.md`

---

### Task 1: Exception hierarchy

**Files:**
- Create: `validation_pipeline/errors.py`
- Test: `tests/test_errors.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_errors.py
from validation_pipeline.errors import (
    PipelineError, DatasetError, LLMError, ToolError,
    CalibrationError, SpecValidationError,
)


def test_pipeline_error_has_module_and_context():
    err = PipelineError("something broke", module="executor", context={"image": "foo.jpg"})
    assert str(err) == "something broke"
    assert err.module == "executor"
    assert err.context == {"image": "foo.jpg"}


def test_pipeline_error_default_context():
    err = PipelineError("fail", module="test")
    assert err.context == {}


def test_tool_error_is_pipeline_error():
    err = ToolError("API timeout", module="nvidia_grounding_dino", context={"http_status": 500})
    assert isinstance(err, PipelineError)
    assert err.module == "nvidia_grounding_dino"


def test_llm_error_is_pipeline_error():
    err = LLMError("Invalid response", module="spec_generator")
    assert isinstance(err, PipelineError)


def test_dataset_error_is_pipeline_error():
    err = DatasetError("Download failed", module="dataset_resolver", context={"url": "http://x.com"})
    assert isinstance(err, PipelineError)


def test_calibration_error_is_pipeline_error():
    err = CalibrationError("No exemplars", module="calibrator")
    assert isinstance(err, PipelineError)


def test_spec_validation_error_is_pipeline_error():
    err = SpecValidationError("Plan not approved", module="compiler")
    assert isinstance(err, PipelineError)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_errors.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement errors.py**

```python
# validation_pipeline/errors.py

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_errors.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/errors.py tests/test_errors.py
git commit -m "feat: add typed exception hierarchy for pipeline errors"
```

---

### Task 2: Shared retry policy

**Files:**
- Create: `validation_pipeline/retry.py`
- Modify: `validation_pipeline/config.py:4-11`
- Test: `tests/test_retry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_retry.py
import time
from unittest.mock import patch
from validation_pipeline.retry import retry_with_policy
from validation_pipeline.config import RetryPolicy
from validation_pipeline.errors import ToolError


def test_retry_succeeds_first_try():
    result = retry_with_policy(
        fn=lambda: "ok",
        policy=RetryPolicy(),
        error_cls=ToolError,
        module="test",
    )
    assert result == "ok"


def test_retry_succeeds_after_transient_failure():
    call_count = {"n": 0}
    def flaky():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("transient")
        return "recovered"

    with patch("time.sleep"):
        result = retry_with_policy(
            fn=flaky,
            policy=RetryPolicy(max_retries=3),
            error_cls=ToolError,
            module="test",
        )
    assert result == "recovered"
    assert call_count["n"] == 2


def test_retry_raises_on_exhaustion():
    def always_fails():
        raise TimeoutError("timeout")

    import pytest
    with patch("time.sleep"):
        with pytest.raises(ToolError) as exc_info:
            retry_with_policy(
                fn=always_fails,
                policy=RetryPolicy(max_retries=2),
                error_cls=ToolError,
                module="nvidia_grounding_dino",
                context={"target_label": "horse"},
            )
    assert exc_info.value.module == "nvidia_grounding_dino"
    assert "retry_count" in exc_info.value.context
    assert exc_info.value.context["retry_count"] == 2


def test_retry_raises_immediately_on_permanent_error():
    """HTTP 401/403 should not be retried."""
    from requests.exceptions import HTTPError
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.status_code = 401

    call_count = {"n": 0}
    def auth_fail():
        call_count["n"] += 1
        err = HTTPError(response=resp)
        raise err

    import pytest
    with patch("time.sleep"):
        with pytest.raises(ToolError):
            retry_with_policy(
                fn=auth_fail,
                policy=RetryPolicy(max_retries=3),
                error_cls=ToolError,
                module="test",
            )
    assert call_count["n"] == 1  # No retries


def test_retry_exponential_backoff():
    """Verify sleep is called with exponential delays."""
    sleep_calls = []
    def always_fails():
        raise ConnectionError("fail")

    import pytest
    with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        with pytest.raises(ToolError):
            retry_with_policy(
                fn=always_fails,
                policy=RetryPolicy(max_retries=3, base_delay=1.0, backoff_factor=2.0),
                error_cls=ToolError,
                module="test",
            )
    assert sleep_calls == [1.0, 2.0, 4.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_retry.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Add RetryPolicy to config.py**

In `validation_pipeline/config.py`, add `RetryPolicy` before `PipelineConfig`:

```python
from pydantic import BaseModel


class RetryPolicy(BaseModel):
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0


class PipelineConfig(BaseModel):
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""
    device: str = "cpu"
    tool_configs_dir: str = "validation_pipeline/tools/configs"
    default_sample_rate: float = 0.05
    max_retries: int = 3
    retry_policy: RetryPolicy = RetryPolicy()


def load_config(**overrides) -> PipelineConfig:
    return PipelineConfig(**overrides)
```

- [ ] **Step 4: Implement retry.py**

```python
# validation_pipeline/retry.py
import time
from typing import TypeVar, Callable
from requests.exceptions import HTTPError
from validation_pipeline.errors import PipelineError

T = TypeVar("T")

# HTTP status codes that should not be retried
PERMANENT_STATUS_CODES = {400, 401, 403, 404, 422}

# Exception types that are transient and should be retried
TRANSIENT_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)


def _is_permanent_http_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError) and exc.response is not None:
        return exc.response.status_code in PERMANENT_STATUS_CODES
    return False


def retry_with_policy(
    fn: Callable[[], T],
    policy,
    error_cls: type[PipelineError],
    module: str,
    context: dict | None = None,
) -> T:
    """
    Call fn() with retries according to policy.
    Retries on transient failures. Raises immediately on permanent failures.
    On exhaustion, raises error_cls with full context.
    """
    context = context or {}
    last_exception = None

    for attempt in range(policy.max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e

            # Permanent errors: raise immediately
            if _is_permanent_http_error(e):
                raise error_cls(
                    f"Permanent error: {e}",
                    module=module,
                    context={**context, "retry_count": attempt, "error_type": type(e).__name__},
                ) from e

            # Last attempt: raise
            if attempt == policy.max_retries:
                raise error_cls(
                    f"Retry exhausted after {policy.max_retries + 1} attempts: {e}",
                    module=module,
                    context={**context, "retry_count": policy.max_retries, "error_type": type(e).__name__},
                ) from e

            # Transient: sleep and retry
            delay = min(
                policy.base_delay * (policy.backoff_factor ** attempt),
                policy.max_delay,
            )
            time.sleep(delay)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_retry.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add validation_pipeline/retry.py validation_pipeline/config.py tests/test_retry.py
git commit -m "feat: add shared retry policy with exponential backoff"
```

---

### Task 3: Schema changes — ImageResult.errors + ExecutionSummary.error_count

**Files:**
- Modify: `validation_pipeline/schemas/execution.py:17-24,27-35`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_schemas.py
from validation_pipeline.schemas.execution import ImageResult, ExecutionSummary


def test_image_result_has_errors_field():
    ir = ImageResult(
        image_id="test", image_path="/test.jpg",
        tool_results=[], verdict="error",
        verdict_reason="API failed",
        errors=["nvidia_grounding_dino: API timeout"],
    )
    assert ir.errors == ["nvidia_grounding_dino: API timeout"]
    assert ir.verdict == "error"


def test_image_result_errors_default_empty():
    ir = ImageResult(
        image_id="test", image_path="/test.jpg",
        tool_results=[], verdict="usable",
        verdict_reason="All passed",
    )
    assert ir.errors == []


def test_execution_summary_has_error_count():
    summary = ExecutionSummary(error_count=3)
    assert summary.error_count == 3


def test_execution_summary_error_count_default_zero():
    summary = ExecutionSummary()
    assert summary.error_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_schemas.py::test_image_result_has_errors_field tests/test_schemas.py::test_execution_summary_has_error_count -v`
Expected: FAIL — fields don't exist

- [ ] **Step 3: Add errors to ImageResult and error_count to ExecutionSummary**

In `validation_pipeline/schemas/execution.py`:

Add `errors: list[str] = []` to `ImageResult` (after line 23, the `exemplar_similarity` field).

Add `error_count: int = 0` to `ExecutionSummary` (after line 30, the `unusable_count` field).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_schemas.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add validation_pipeline/schemas/execution.py tests/test_schemas.py
git commit -m "feat: add errors field to ImageResult, error_count to ExecutionSummary"
```

---

### Task 4: Tool wrappers — replace silent fallbacks with ToolError

**Files:**
- Modify: `validation_pipeline/tools/wrappers/nvidia_nim_wrapper.py:90-102`
- Modify: `validation_pipeline/tools/wrappers/roboflow_wrapper.py:26-49`
- Modify: `validation_pipeline/tools/wrappers/openai_vision_wrapper.py:40-72`
- Test: `tests/test_tools/test_nvidia_nim_wrapper.py`, `tests/test_tools/test_roboflow_wrapper.py`, `tests/test_tools/test_openai_vision_wrapper.py`

- [ ] **Step 1: Write failing tests for ToolError on exhaustion**

```python
# Append to tests/test_tools/test_nvidia_nim_wrapper.py
import pytest
from validation_pipeline.errors import ToolError


def test_nvidia_dino_raises_tool_error_on_exhaustion():
    tool = _make_tool()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    with patch("requests.post", side_effect=Exception("always fails")), \
         patch.dict("os.environ", {"NVIDIA_NIM_API_KEY": "nvapi-test"}), \
         patch("time.sleep"):
        with pytest.raises(ToolError) as exc_info:
            tool.execute(img, target_label="horse")
    assert exc_info.value.module == "nvidia_grounding_dino"
```

```python
# Append to tests/test_tools/test_roboflow_wrapper.py
import pytest
from validation_pipeline.errors import ToolError


def test_roboflow_raises_tool_error_on_exhaustion():
    tool = RoboflowObjectDetectionTool({"api_key_env": "ROBOFLOW_API_KEY", "model": "coco/1"})
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    with patch("requests.post", side_effect=Exception("always fails")), \
         patch.dict("os.environ", {"ROBOFLOW_API_KEY": "test-key"}), \
         patch("time.sleep"):
        with pytest.raises(ToolError) as exc_info:
            tool.execute(img, target_label="horse")
    assert exc_info.value.module == "roboflow_object_detection"
```

```python
# Append to tests/test_tools/test_openai_vision_wrapper.py
import pytest
from validation_pipeline.errors import ToolError


def test_gpt4v_raises_tool_error_on_exhaustion():
    tool = GPT4VisionTool({"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o", "max_tokens": 100})
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("always fails")

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
         patch("instructor.from_openai", return_value=mock_client), \
         patch("time.sleep"):
        with pytest.raises(ToolError) as exc_info:
            tool.execute(img, semantic_question="test")
    assert exc_info.value.module == "gpt4o_vision_semantic"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_tools/test_nvidia_nim_wrapper.py::test_nvidia_dino_raises_tool_error_on_exhaustion tests/test_tools/test_roboflow_wrapper.py::test_roboflow_raises_tool_error_on_exhaustion tests/test_tools/test_openai_vision_wrapper.py::test_gpt4v_raises_tool_error_on_exhaustion -v`
Expected: FAIL — wrappers return dict instead of raising

- [ ] **Step 3: Update NIM wrapper**

In `validation_pipeline/tools/wrappers/nvidia_nim_wrapper.py`, replace the `execute()` method (lines 90-102):

```python
    def execute(self, image: Image.Image, **kwargs) -> dict:
        target_label = kwargs.get("target_label", "")
        b64_image = self._encode_image(image)

        from validation_pipeline.retry import retry_with_policy
        from validation_pipeline.config import RetryPolicy
        from validation_pipeline.errors import ToolError

        def _call():
            result = self._run_inference(b64_image, target_label)
            result["target_label"] = target_label
            return result

        return retry_with_policy(
            fn=_call,
            policy=RetryPolicy(),
            error_cls=ToolError,
            module=self.name,
            context={"target_label": target_label},
        )
```

Remove `import time` from the top of the file (no longer needed). Remove `self.max_retries` from `__init__` (retries handled by policy). Keep `self.timeout` — it's still used by `_run_inference` for the HTTP request timeout.

- [ ] **Step 4: Update Roboflow wrapper**

In `validation_pipeline/tools/wrappers/roboflow_wrapper.py`, replace the `execute()` method (lines 26-59):

```python
    def execute(self, image: Image.Image, **kwargs) -> dict:
        target_label = kwargs.get("target_label", "")
        api_key = os.environ.get(self.api_key_env, "")

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        from validation_pipeline.retry import retry_with_policy
        from validation_pipeline.config import RetryPolicy
        from validation_pipeline.errors import ToolError

        def _call():
            url = f"https://detect.roboflow.com/{self.model}"
            params = {"api_key": api_key}
            resp = requests.post(
                url, params=params, files={"file": ("image.jpg", img_bytes)},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            detections = data.get("predictions", [])
            matching = [d for d in detections if d.get("class", "").lower() == target_label.lower()]
            best_confidence = max((d["confidence"] for d in matching), default=0.0)
            return {
                "best_confidence": best_confidence,
                "detections": detections,
                "target_label": target_label,
            }

        return retry_with_policy(
            fn=_call,
            policy=RetryPolicy(),
            error_cls=ToolError,
            module=self.name,
            context={"target_label": target_label},
        )
```

Remove `import time` from imports. Remove `self.max_retries` from `__init__`.

- [ ] **Step 5: Update GPT-4o Vision wrapper**

In `validation_pipeline/tools/wrappers/openai_vision_wrapper.py`, replace the `execute()` method (lines 40-72):

```python
    def execute(self, image: Image.Image, **kwargs) -> dict:
        semantic_question = kwargs.get("semantic_question", "overall quality")
        api_key = os.environ.get(self.api_key_env, "")
        b64_image = self._encode_image(image)

        from validation_pipeline.retry import retry_with_policy
        from validation_pipeline.config import RetryPolicy
        from validation_pipeline.errors import ToolError

        def _call():
            client = instructor.from_openai(OpenAI(api_key=api_key))
            result = client.chat.completions.create(
                model=self.model,
                response_model=VLMResult,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            f"Rate this image on: \"{semantic_question}\". "
                            "Return a score from 0.0 (worst) to 1.0 (best) "
                            "and a one-sentence justification."
                        )},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }},
                    ]},
                ],
            )
            return {
                "score": result.score,
                "justification": result.justification,
                "semantic_question": semantic_question,
            }

        return retry_with_policy(
            fn=_call,
            policy=RetryPolicy(),
            error_cls=ToolError,
            module=self.name,
            context={"semantic_question": semantic_question},
        )
```

Remove `import time` from imports. Remove `self.max_retries` from `__init__`. Keep `self.timeout`.

- [ ] **Step 6: Update existing retry tests**

The existing `test_roboflow_retry_on_failure`, `test_nvidia_dino_retry_on_failure`, and `test_gpt4v_retry_on_failure` tests need to be updated since retry is now handled by `retry_with_policy`. These tests should mock `retry_with_policy` or test through it. Simplest: remove the old retry tests since retry behavior is now tested in `tests/test_retry.py`, and keep the new `raises_tool_error_on_exhaustion` tests.

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_tools/ -v`
Expected: ALL PASS

- [ ] **Step 8: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add validation_pipeline/tools/wrappers/ tests/test_tools/
git commit -m "feat: tool wrappers use retry_with_policy, raise ToolError on failure"
```

---

### Task 5: LLM modules — wrap instructor calls with LLMError

**Files:**
- Modify: `validation_pipeline/modules/spec_generator.py:27-46`
- Modify: `validation_pipeline/modules/planner.py:22-52`
- Modify: `validation_pipeline/modules/dataset_resolver.py:31-42`
- Test: `tests/test_spec_generator.py`, `tests/test_planner.py`, `tests/test_dataset_resolver.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_spec_generator.py
import pytest
from unittest.mock import patch
from validation_pipeline.errors import LLMError
from validation_pipeline.modules.spec_generator import generate_spec
from validation_pipeline.schemas.user_input import UserInput


def test_spec_generator_raises_llm_error():
    ui = UserInput(dataset_path="/data", intent="test")
    with patch("validation_pipeline.modules.spec_generator._call_llm", side_effect=Exception("LLM failed")):
        with pytest.raises(LLMError) as exc_info:
            generate_spec(ui)
    assert exc_info.value.module == "spec_generator"
```

```python
# Append to tests/test_planner.py
import pytest
from unittest.mock import patch
from validation_pipeline.errors import LLMError
from validation_pipeline.modules.planner import generate_plan
from validation_pipeline.schemas.spec import FormalSpec, QualityCriterion, QuantityTarget, OutputFormat
from validation_pipeline.schemas.calibration import CalibrationResult


def test_planner_raises_llm_error():
    spec = FormalSpec(
        restated_request="test", assumptions=[], content_criteria=[],
        quality_criteria=[], quantity_targets=QuantityTarget(),
        output_format=OutputFormat(), success_criteria="test",
    )
    cal = CalibrationResult(tool_calibrations={}, exemplar_embeddings=[], threshold_report=[])
    with patch("validation_pipeline.modules.planner._call_llm", side_effect=Exception("LLM failed")):
        with pytest.raises(LLMError) as exc_info:
            generate_plan(spec, cal, [])
    assert exc_info.value.module == "planner"
```

```python
# Append to tests/test_dataset_resolver.py
# Add these imports at the top if not already present:
# import pytest
# from unittest.mock import patch
# from validation_pipeline.errors import LLMError, DatasetError
# from validation_pipeline.schemas.dataset import DatasetPlan
import pytest
from validation_pipeline.errors import LLMError, DatasetError
from validation_pipeline.modules.dataset_resolver import resolve_dataset, download_dataset


def test_resolve_dataset_raises_llm_error():
    with patch("validation_pipeline.modules.dataset_resolver._call_llm", side_effect=Exception("LLM failed")):
        with pytest.raises(LLMError) as exc_info:
            resolve_dataset("test")
    assert exc_info.value.module == "dataset_resolver"


def test_download_dataset_raises_dataset_error_unknown_source():
    plan = DatasetPlan(
        source="unknown", url="http://x.com",
        max_images=10, download_path="/tmp/test",
    )
    with pytest.raises(DatasetError) as exc_info:
        download_dataset(plan)
    assert exc_info.value.module == "dataset_resolver"
    assert "unknown" in str(exc_info.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_spec_generator.py::test_spec_generator_raises_llm_error tests/test_planner.py::test_planner_raises_llm_error tests/test_dataset_resolver.py::test_resolve_dataset_raises_llm_error -v`
Expected: FAIL — modules don't raise LLMError

- [ ] **Step 3: Update spec_generator.py**

In `validation_pipeline/modules/spec_generator.py`, update `generate_spec`:

```python
def generate_spec(user_input: UserInput, config: PipelineConfig | None = None) -> FormalSpec:
    from validation_pipeline.errors import LLMError
    try:
        spec = _call_llm(user_input, config)
    except Exception as e:
        raise LLMError(
            f"Spec generation failed: {e}",
            module="spec_generator",
            context={"intent": user_input.intent},
        ) from e
    spec.user_confirmed = False
    return spec
```

- [ ] **Step 4: Update planner.py**

In `validation_pipeline/modules/planner.py`, update `generate_plan`:

```python
def generate_plan(
    spec: FormalSpec,
    calibration: CalibrationResult,
    tools: list[dict],
    config: PipelineConfig | None = None,
) -> ValidationPlan:
    from validation_pipeline.errors import LLMError
    try:
        plan = _call_llm(spec, calibration, tools, config)
    except Exception as e:
        raise LLMError(
            f"Plan generation failed: {e}",
            module="planner",
            context={"spec_summary": spec.restated_request},
        ) from e
    plan.user_approved = False
    return plan
```

- [ ] **Step 5: Update dataset_resolver.py**

In `validation_pipeline/modules/dataset_resolver.py`, update `resolve_dataset` and `download_dataset`:

```python
def resolve_dataset(description: str, config: PipelineConfig | None = None) -> DatasetPlan:
    from validation_pipeline.errors import LLMError
    try:
        return _call_llm(description, config)
    except Exception as e:
        raise LLMError(
            f"Dataset resolution failed: {e}",
            module="dataset_resolver",
            context={"description": description},
        ) from e


def download_dataset(plan: DatasetPlan) -> str:
    from validation_pipeline.errors import DatasetError
    factory = DOWNLOADERS.get(plan.source)
    if not factory:
        raise DatasetError(
            f"Unknown dataset source: {plan.source}. Supported: {list(DOWNLOADERS.keys())}",
            module="dataset_resolver",
            context={"source": plan.source, "supported": list(DOWNLOADERS.keys())},
        )
    downloader = factory()
    try:
        return downloader.download(plan)
    except Exception as e:
        raise DatasetError(
            f"Dataset download failed: {e}",
            module="dataset_resolver",
            context={"source": plan.source, "url": plan.url},
        ) from e
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_spec_generator.py tests/test_planner.py tests/test_dataset_resolver.py -v`
Expected: ALL PASS

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add validation_pipeline/modules/spec_generator.py validation_pipeline/modules/planner.py validation_pipeline/modules/dataset_resolver.py tests/test_spec_generator.py tests/test_planner.py tests/test_dataset_resolver.py
git commit -m "feat: LLM modules raise LLMError, dataset resolver raises DatasetError"
```

---

### Task 6: Calibrator — graceful no-exemplar handling + error collection

**Files:**
- Modify: `validation_pipeline/modules/calibrator.py:11-46`
- Test: `tests/test_calibrator.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_calibrator.py
from validation_pipeline.schemas.spec import FormalSpec, QualityCriterion, QuantityTarget, OutputFormat


def test_calibrator_no_exemplars_returns_empty_result():
    """With no exemplars, calibrator should return clean empty result without numpy warnings."""
    spec = FormalSpec(
        restated_request="test", assumptions=[],
        content_criteria=[],
        quality_criteria=[QualityCriterion(dimension="blur", description="sharp")],
        quantity_targets=QuantityTarget(),
        output_format=OutputFormat(),
        success_criteria="test",
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = calibrate(spec, [], [], {})
    assert result.tool_calibrations == {}
    assert len(result.threshold_report) == 1  # One per quality criterion
    assert "No exemplars" in result.threshold_report[0].explanation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_calibrator.py::test_calibrator_no_exemplars_returns_empty_result -v`
Expected: FAIL — numpy warnings raised because `np.mean([])` is called

- [ ] **Step 3: Update calibrator to handle no exemplars gracefully**

In `validation_pipeline/modules/calibrator.py`, add an early return at the top of `calibrate()` (after line 18):

```python
    # No exemplars: skip calibration entirely
    if not good_paths and not bad_paths:
        return CalibrationResult(
            tool_calibrations={},
            exemplar_embeddings=[],
            threshold_report=[
                ThresholdExplanation(
                    dimension=qc.dimension,
                    threshold=0.5,
                    explanation="No exemplars provided. Using default thresholds.",
                )
                for qc in spec.quality_criteria
            ],
        )
```

Also wrap the tool execution in a try/except to skip failing images:

Replace lines 26-27 (the list comprehensions):
```python
        good_scores = []
        for p in good_paths:
            try:
                good_scores.append(tool.execute(Image.open(p).convert("RGB")))
            except Exception:
                pass  # Skip failing images

        bad_scores = []
        for p in bad_paths:
            try:
                bad_scores.append(tool.execute(Image.open(p).convert("RGB")))
            except Exception:
                pass  # Skip failing images

        if not good_scores and not bad_scores:
            from validation_pipeline.errors import CalibrationError
            raise CalibrationError(
                f"All exemplar images failed for dimension '{dim}'",
                module="calibrator",
                context={"dimension": dim, "good_count": len(good_paths), "bad_count": len(bad_paths)},
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_calibrator.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add validation_pipeline/modules/calibrator.py tests/test_calibrator.py
git commit -m "feat: calibrator handles no-exemplar case gracefully, skips failing images"
```

---

### Task 7: Compiler — use SpecValidationError

**Files:**
- Modify: `validation_pipeline/modules/compiler.py:7-8`
- Test: `tests/test_compiler.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_compiler.py
import pytest
from validation_pipeline.errors import SpecValidationError


def test_compile_unapproved_plan_raises_spec_validation_error():
    plan = ValidationPlan(
        plan_id="p1", spec_summary="test",
        sampling_strategy=SamplingStrategy(),
        steps=[], combination_logic="ALL_PASS",
        estimated_cost=CostEstimate(),
        user_approved=False,
    )
    with pytest.raises(SpecValidationError) as exc_info:
        compile_plan(plan)
    assert exc_info.value.module == "compiler"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_compiler.py::test_compile_unapproved_plan_raises_spec_validation_error -v`
Expected: FAIL — raises ValueError, not SpecValidationError

- [ ] **Step 3: Update compiler**

In `validation_pipeline/modules/compiler.py`, change line 8:

```python
# Before:
        raise ValueError("Cannot compile a plan that is not approved by the user")
# After:
    from validation_pipeline.errors import SpecValidationError
    if not plan.user_approved:
        raise SpecValidationError(
            "Cannot compile a plan that is not approved by the user",
            module="compiler",
            context={"plan_id": plan.plan_id},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_compiler.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/modules/compiler.py tests/test_compiler.py
git commit -m "feat: compiler raises SpecValidationError instead of ValueError"
```

---

### Task 8: Executor — error collection + "error" verdict

**Files:**
- Modify: `validation_pipeline/modules/executor.py:33-46,60-135,138-157`
- Test: `tests/test_executor.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_executor.py
from validation_pipeline.errors import ToolError


def test_executor_collects_tool_errors(tmp_path):
    """When a tool raises ToolError, executor records it instead of crashing."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img_dir / "test.jpg"))

    class FailingTool(BaseTool):
        name = "failing_tool"
        task_type = "content_detection"
        tier = 2
        def execute(self, image, **kwargs):
            raise ToolError("API timeout", module="failing_tool", context={"http_status": 500})
        def normalize(self, raw_output, calibration=None):
            pass

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[ProgramLine(
            line_number=1, variable_name="content_score",
            tool_call="failing_tool(image)", output_type="float",
            threshold_check="content_score >= 0.5", tier=2,
        )],
        batch_strategy=BatchStrategy(),
        tool_imports=["failing_tool"],
    )
    tools = {"failing_tool": FailingTool({})}
    result = execute_program(program, str(img_dir), tools)

    assert result.results[0].verdict == "error"
    assert len(result.results[0].errors) > 0
    assert "API timeout" in result.results[0].errors[0]
    assert result.summary.error_count == 1


def test_executor_partial_tool_failure(tmp_path):
    """When some tools succeed and some fail, verdict uses available results."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img_dir / "test.jpg"))

    class GoodTool(BaseTool):
        name = "good_tool"
        task_type = "image_quality"
        tier = 1
        def execute(self, image, **kwargs):
            return 0.9
        def normalize(self, raw_output, calibration=None):
            return ToolResult(
                tool_name="good_tool", dimension="blur",
                score=0.9, passed=True, threshold=0.5, raw_output=0.9,
            )

    class BadTool(BaseTool):
        name = "bad_tool"
        task_type = "content_detection"
        tier = 2
        def execute(self, image, **kwargs):
            raise ToolError("timeout", module="bad_tool")
        def normalize(self, raw_output, calibration=None):
            pass

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[
            ProgramLine(line_number=1, variable_name="blur_score", tool_call="good_tool(image)", output_type="float", threshold_check="blur_score >= 0.5", tier=1),
            ProgramLine(line_number=2, variable_name="content_score", tool_call="bad_tool(image)", output_type="float", threshold_check="content_score >= 0.5", tier=2),
        ],
        batch_strategy=BatchStrategy(early_exit=False),
        tool_imports=["good_tool", "bad_tool"],
    )
    tools = {"good_tool": GoodTool({}), "bad_tool": BadTool({})}
    result = execute_program(program, str(img_dir), tools)

    # Should use available results, not verdict="error"
    img = result.results[0]
    assert len(img.errors) > 0  # bad_tool error recorded
    assert len(img.tool_results) == 1  # good_tool result present
    assert img.verdict != "error"  # partial failure, not total
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_executor.py::test_executor_collects_tool_errors tests/test_executor.py::test_executor_partial_tool_failure -v`
Expected: FAIL — executor crashes or returns "unusable" instead of "error"

- [ ] **Step 3: Update executor**

In `validation_pipeline/modules/executor.py`, make these changes:

**Change 1**: Update `_run_program_on_image` to catch ToolError per-tool (around line 88-91):

Replace lines 88-91:
```python
        tool = tools[tool_name]
        params = line.tool_params or {}
        raw_output = tool.execute(image, **params)
        lines_executed += 1
```

With:
```python
        tool = tools[tool_name]
        params = line.tool_params or {}
        try:
            raw_output = tool.execute(image, **params)
        except Exception as e:
            errors.append(f"{tool_name}: {e}")
            continue
        lines_executed += 1
```

Add `errors = []` at line 70 (alongside `reasons = []`).

**Change 2**: Update verdict logic (around line 118-126):

Replace:
```python
    if all_passed:
        verdict = "usable"
        reason = "All checks passed"
    elif len(reasons) == 1:
        verdict = "recoverable"
        reason = "; ".join(reasons)
    else:
        verdict = "unusable"
        reason = "; ".join(reasons)
```

With:
```python
    if not tool_results and errors:
        verdict = "error"
        reason = "All tools failed: " + "; ".join(errors)
    elif all_passed:
        verdict = "usable"
        reason = "All checks passed"
    elif len(reasons) == 1:
        verdict = "recoverable"
        reason = "; ".join(reasons)
    else:
        verdict = "unusable"
        reason = "; ".join(reasons)
```

**Change 3**: Pass errors to ImageResult (line 128-135):

Add `errors=errors,` to the ImageResult constructor.

**Change 4**: Update outer exception handler (lines 38-46):

Replace `verdict="unusable"` with `verdict="error"` and add `errors=[str(e)]`.

**Change 5**: Update `_compute_summary` (lines 138-157):

Add `error_count` and populate `tool_error_rate` from per-image errors:
```python
    error = sum(1 for r in results if r.verdict == "error")

    # Populate tool_error_rate from per-image errors
    tool_errors: dict[str, int] = {}
    for r in results:
        for err_msg in r.errors:
            tool_name = err_msg.split(":")[0].strip() if ":" in err_msg else "unknown"
            tool_errors[tool_name] = tool_errors.get(tool_name, 0) + 1
    tool_error_rate = {k: v / total for k, v in tool_errors.items()}
```
And add `error_count=error, tool_error_rate=tool_error_rate,` to the ExecutionSummary constructor.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_executor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add validation_pipeline/modules/executor.py tests/test_executor.py
git commit -m "feat: executor collects per-image errors, adds 'error' verdict"
```

---

### Task 9: Supervisor + Reporter — handle error verdict

**Files:**
- Modify: `validation_pipeline/modules/supervisor.py:44-57`
- Modify: `validation_pipeline/modules/reporter.py:18-22`
- Modify: `validation_pipeline/schemas/report.py:5-11`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_supervisor.py
from validation_pipeline.schemas.execution import ExecutionResult, ExecutionSummary


def test_supervisor_detects_high_error_rate():
    result = ExecutionResult(
        phase="full", total_images=10, processed=10,
        summary=ExecutionSummary(
            usable_count=2, recoverable_count=1, unusable_count=2,
            error_count=5,
        ),
    )
    from validation_pipeline.schemas.calibration import CalibrationResult
    from validation_pipeline.schemas.plan import ValidationPlan, SamplingStrategy, CostEstimate
    cal = CalibrationResult(tool_calibrations={}, exemplar_embeddings=[], threshold_report=[])
    plan = ValidationPlan(
        plan_id="p1", spec_summary="test", sampling_strategy=SamplingStrategy(),
        steps=[], combination_logic="ALL_PASS", estimated_cost=CostEstimate(), user_approved=True,
    )
    report = supervise(result, cal, plan)
    error_checks = [c for c in report.checks if c.check_name == "image_error_rate"]
    assert len(error_checks) == 1
    assert not error_checks[0].passed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_supervisor.py::test_supervisor_detects_high_error_rate -v`
Expected: FAIL — supervisor doesn't check error_count

- [ ] **Step 3: Update supervisor to detect high error rates**

In `validation_pipeline/modules/supervisor.py`, add after the tool_error_rate loop (after line 43):

```python
    # Check for high image error rate
    error_count = result.summary.error_count
    if result.processed > 0:
        error_rate = error_count / result.total_images
        checks.append(SupervisionCheck(
            check_name="image_error_rate",
            passed=error_rate < 0.1,
            details=f"Image error rate: {error_rate:.1%} ({error_count}/{result.total_images})",
        ))
        if error_rate >= 0.1:
            anomalies.append(Anomaly(
                severity="blocker" if error_rate > 0.5 else "warning",
                description=f"{error_count} images failed with tool errors ({error_rate:.0%})",
                likely_cause="Tool API failures or incompatible image formats",
                suggested_action="Check API keys, network connectivity, and image formats",
            ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_supervisor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Add error_count to DatasetStats**

In `validation_pipeline/schemas/report.py`, add to `DatasetStats`:
```python
    error_count: int = 0
```

- [ ] **Step 6: Update reporter to include error_count**

In `validation_pipeline/modules/reporter.py`, add `error_count`:

After line 21 (`unusable = result.summary.unusable_count`), add:
```python
    error_count = result.summary.error_count
```

Add `error_count=error_count,` to the `DatasetStats(...)` constructor.

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add validation_pipeline/modules/supervisor.py validation_pipeline/modules/reporter.py validation_pipeline/schemas/report.py tests/test_supervisor.py
git commit -m "feat: supervisor detects error rate anomalies, reporter tracks error_count"
```

---

### Task 10: Pipeline orchestrator — typed exception handling

**Files:**
- Modify: `validation_pipeline/pipeline.py:21-97`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_pipeline.py
# Add import: from validation_pipeline.errors import DatasetError, SpecValidationError
import pytest


def test_pipeline_raises_dataset_error_no_path_or_description():
    config = PipelineConfig(tool_configs_dir="/nonexistent")
    pipeline = ValidationPipeline(config)
    with pytest.raises(DatasetError):
        pipeline.run(UserInput(intent="test"), auto_approve=True)


def test_pipeline_raises_spec_validation_error_unconfirmed(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = PipelineConfig(tool_configs_dir=str(config_dir))
    pipeline = ValidationPipeline(config)
    mock_spec = FormalSpec(
        restated_request="test", assumptions=[], content_criteria=[],
        quality_criteria=[], quantity_targets=QuantityTarget(),
        output_format=OutputFormat(), success_criteria="test",
        user_confirmed=False,
    )
    with patch("validation_pipeline.modules.spec_generator._call_llm", return_value=mock_spec):
        with pytest.raises(SpecValidationError):
            pipeline.run(UserInput(dataset_path=str(img_dir), intent="test"), auto_approve=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_pipeline.py::test_pipeline_raises_dataset_error_no_path_or_description tests/test_pipeline.py::test_pipeline_raises_spec_validation_error_unconfirmed -v`
Expected: FAIL — raises ValueError, not typed exceptions

- [ ] **Step 3: Update pipeline.py**

In `validation_pipeline/pipeline.py`, add import:
```python
from validation_pipeline.errors import PipelineError, DatasetError, LLMError, SpecValidationError
```

Replace the `ValueError` raises with typed exceptions:

Line 29 (`raise ValueError("Either dataset_path...")`):
```python
                raise DatasetError(
                    "Either dataset_path or dataset_description must be provided",
                    module="pipeline",
                )
```

Line 36 (`raise ValueError("Spec must be confirmed...")`):
```python
            raise SpecValidationError(
                "Spec must be confirmed by user before proceeding",
                module="pipeline",
            )
```

Line 73 (`raise ValueError("Plan must be approved...")`):
```python
            raise SpecValidationError(
                "Plan must be approved by user before proceeding",
                module="pipeline",
            )
```

Line 87 (the `print(f"WARNING: ...")`):
```python
        if unavailable:
            import sys
            print(f"WARNING: Tools not available (skipping): {unavailable}", file=sys.stderr)
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
git commit -m "feat: pipeline uses typed exceptions instead of ValueError"
```

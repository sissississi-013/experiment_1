# Adaptive Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded threshold=0.5 calibration with post-execution batch-adaptive calibration that analyzes score distributions to find natural quality breaks.

**Architecture:** Move calibration after execution. Executor collects all scores without threshold gating. New Recalibrator module analyzes batch score distributions (GMM for multimodal, percentile for unimodal) and assigns verdicts. Planner sets strictness hints (0.0–1.0) instead of hard thresholds.

**Tech Stack:** Python 3.14, Pydantic v2, scikit-learn (GaussianMixture), scipy, numpy, diptest (new ~10KB dep)

**Spec:** `docs/specs/2026-03-22-adaptive-calibration-design.md`

---

### Task 1: Add `diptest` dependency

**Files:**
- Modify: `pyproject.toml` (or `requirements.txt` — check which exists)

- [ ] **Step 1: Check dependency file and add diptest**

```bash
# Check which dependency file exists
ls pyproject.toml requirements.txt 2>/dev/null
```

Add `diptest` to the dependencies list. It's a ~10KB package for the Hartigan dip test.

- [ ] **Step 2: Install and verify**

```bash
pip install diptest
python -c "import diptest; print(diptest.diptest([1,2,3,4,5,6,7,8,9,10]))"
```

Expected: prints a tuple `(dip_statistic, p_value)`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml  # or requirements.txt
git commit -m "chore: add diptest dependency for Hartigan dip test"
```

---

### Task 2: Create recalibration schemas

**Files:**
- Create: `validation_pipeline/schemas/recalibration.py`
- Test: `tests/test_schemas.py` (append to existing)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schemas.py`:

```python
from validation_pipeline.schemas.recalibration import (
    DimensionCalibration, ImageVerdictRecord, RecalibrationResult,
)

def test_dimension_calibration_schema():
    dc = DimensionCalibration(
        dimension="blur", method="gmm",
        thresholds=[0.3, 0.6], confidence=0.85,
        dip_test_p=0.01, gmm_means=[0.2, 0.5, 0.8],
        explanation="Found 3 clusters", strictness=0.5,
    )
    assert dc.dimension == "blur"
    assert len(dc.thresholds) == 2
    dumped = dc.model_dump()
    assert "gmm_means" in dumped

def test_image_verdict_record_schema():
    ivr = ImageVerdictRecord(
        image_id="img_001", image_path="/tmp/img_001.jpg",
        verdict="recoverable", scores={"blur": 0.8, "exposure": 0.3},
        failed_dimensions=["exposure"],
        explanation="Failed exposure (0.30 < 0.45)",
    )
    assert ivr.verdict == "recoverable"
    assert ivr.failed_dimensions == ["exposure"]

def test_recalibration_result_schema():
    dc = DimensionCalibration(
        dimension="blur", method="percentile",
        thresholds=[0.2, 0.5], confidence=0.6,
        dip_test_p=0.3, gmm_means=[],
        explanation="Unimodal distribution", strictness=0.5,
    )
    ivr = ImageVerdictRecord(
        image_id="img_001", image_path="/tmp/img_001.jpg",
        verdict="usable", scores={"blur": 0.8},
        failed_dimensions=[], explanation="All checks passed",
    )
    rr = RecalibrationResult(
        dimension_calibrations={"blur": dc},
        image_verdicts={"img_001": ivr},
        method_summary="1/1 dimensions used percentile",
        overall_confidence=0.6,
    )
    assert rr.overall_confidence == 0.6
    dumped = rr.model_dump()
    assert "dimension_calibrations" in dumped
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_schemas.py::test_dimension_calibration_schema -v
```

Expected: FAIL — `ImportError: cannot import name 'DimensionCalibration' from 'validation_pipeline.schemas.recalibration'`

- [ ] **Step 3: Write the schema file**

Create `validation_pipeline/schemas/recalibration.py`:

```python
from pydantic import BaseModel


class DimensionCalibration(BaseModel):
    dimension: str
    method: str  # "gmm" | "percentile" | "exemplar"
    thresholds: list[float]  # [unusable_upper, recoverable_upper]
    confidence: float  # GVF score (0-1)
    dip_test_p: float  # Hartigan p-value
    gmm_means: list[float] = []
    explanation: str
    strictness: float


class ImageVerdictRecord(BaseModel):
    image_id: str
    image_path: str
    verdict: str  # "usable" | "recoverable" | "unusable" | "error"
    scores: dict[str, float]
    failed_dimensions: list[str]
    explanation: str


class RecalibrationResult(BaseModel):
    dimension_calibrations: dict[str, DimensionCalibration]
    image_verdicts: dict[str, ImageVerdictRecord]
    method_summary: str
    overall_confidence: float
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schemas.py -v -k "recalib or dimension_calib or image_verdict_record"
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/schemas/recalibration.py tests/test_schemas.py
git commit -m "feat: add recalibration schemas (DimensionCalibration, ImageVerdictRecord, RecalibrationResult)"
```

---

### Task 3: Build the Recalibrator module (core logic)

This is the core new module. It analyzes score distributions and assigns verdicts.

**Files:**
- Create: `validation_pipeline/modules/recalibrator.py`
- Create: `tests/test_recalibrator.py`

- [ ] **Step 1: Write failing tests for distribution analysis**

Create `tests/test_recalibrator.py`:

```python
import numpy as np
import pytest
from validation_pipeline.modules.recalibrator import (
    _analyze_dimension, _compute_gvf, _assign_verdicts, recalibrate,
)
from validation_pipeline.schemas.recalibration import DimensionCalibration
from validation_pipeline.schemas.execution import ExecutionResult, ImageResult, ToolResult


def _make_bimodal_scores(n=50):
    """Two clear Gaussians: low cluster ~0.2, high cluster ~0.8."""
    rng = np.random.RandomState(42)
    low = rng.normal(0.2, 0.05, n // 2).clip(0, 1)
    high = rng.normal(0.8, 0.05, n - n // 2).clip(0, 1)
    return list(np.concatenate([low, high]))


def _make_uniform_scores(n=50):
    rng = np.random.RandomState(42)
    return list(rng.uniform(0, 1, n))


def _make_identical_scores(n=20, val=0.5):
    return [val] * n


def test_analyze_bimodal_uses_gmm():
    scores = _make_bimodal_scores(50)
    dc = _analyze_dimension("blur", scores, strictness=0.5)
    assert dc.method == "gmm"
    assert len(dc.thresholds) == 2
    assert dc.thresholds[0] < dc.thresholds[1]
    assert 0.3 < dc.thresholds[0] < 0.7  # crossing point near middle
    assert dc.confidence > 0.5


def test_analyze_uniform_uses_percentile():
    scores = _make_uniform_scores(50)
    dc = _analyze_dimension("exposure", scores, strictness=0.5)
    assert dc.method == "percentile"
    assert len(dc.thresholds) == 2
    assert dc.thresholds[0] < dc.thresholds[1]


def test_analyze_identical_scores():
    scores = _make_identical_scores(20, 0.5)
    dc = _analyze_dimension("blur", scores, strictness=0.5)
    assert dc.method == "percentile"
    assert "no score variation" in dc.explanation.lower() or dc.confidence < 0.1


def test_analyze_small_batch_uses_percentile():
    scores = [0.3, 0.5, 0.7, 0.8, 0.9, 0.4, 0.6]
    dc = _analyze_dimension("blur", scores, strictness=0.5)
    assert dc.method == "percentile"  # too few for GMM


def test_analyze_tiny_batch_uses_defaults():
    scores = [0.5, 0.6, 0.7]
    dc = _analyze_dimension("blur", scores, strictness=0.5)
    assert dc.method == "percentile"
    assert "small batch" in dc.explanation.lower() or "insufficient" in dc.explanation.lower()


def test_strictness_shifts_percentile_boundaries():
    scores = list(np.linspace(0.0, 1.0, 100))
    lenient = _analyze_dimension("blur", scores, strictness=0.0)
    strict = _analyze_dimension("blur", scores, strictness=1.0)
    # Strict should have higher thresholds (fewer images pass)
    assert strict.thresholds[0] > lenient.thresholds[0]
    assert strict.thresholds[1] > lenient.thresholds[1]


def test_gvf_perfect_separation():
    labels = [0] * 25 + [1] * 25
    scores = [0.1] * 25 + [0.9] * 25
    gvf = _compute_gvf(scores, labels)
    assert gvf > 0.9


def test_gvf_no_separation():
    scores = [0.5] * 50
    labels = [0] * 25 + [1] * 25
    gvf = _compute_gvf(scores, labels)
    assert gvf < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_recalibrator.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `_analyze_dimension`, `_compute_gvf`**

Create `validation_pipeline/modules/recalibrator.py`:

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from validation_pipeline.schemas.recalibration import (
    DimensionCalibration, ImageVerdictRecord, RecalibrationResult,
)
from validation_pipeline.schemas.execution import ExecutionResult
from validation_pipeline.schemas.calibration import CalibrationResult

try:
    import diptest
    HAS_DIPTEST = True
except ImportError:
    HAS_DIPTEST = False


def _compute_gvf(scores: list[float], labels: list[int]) -> float:
    """Goodness of Variance Fit: 1.0 = perfect clustering, 0.0 = no clustering."""
    arr = np.array(scores)
    lab = np.array(labels)
    total_var = np.var(arr)
    if total_var == 0:
        return 0.0
    within_var = sum(
        np.var(arr[lab == k]) * np.sum(lab == k)
        for k in np.unique(lab)
    ) / len(arr)
    return 1.0 - (within_var / total_var)


def _gmm_crossing_points(gmm: GaussianMixture) -> list[float]:
    """Find crossing points between adjacent Gaussian components, sorted by mean."""
    means = gmm.means_.flatten()
    order = np.argsort(means)
    sorted_means = means[order]

    crossings = []
    for i in range(len(sorted_means) - 1):
        # Approximate crossing as weighted midpoint
        m1, m2 = sorted_means[i], sorted_means[i + 1]
        s1 = np.sqrt(gmm.covariances_.flatten()[order[i]])
        s2 = np.sqrt(gmm.covariances_.flatten()[order[i + 1]])
        # Weighted by inverse std — tighter cluster pulls crossing closer
        if s1 + s2 > 0:
            crossing = (m1 * s2 + m2 * s1) / (s1 + s2)
        else:
            crossing = (m1 + m2) / 2
        crossings.append(float(crossing))
    return crossings


def _analyze_dimension(
    dimension: str,
    scores: list[float],
    strictness: float = 0.5,
    exemplar_threshold: float | None = None,
    exemplar_separability: float | None = None,
) -> DimensionCalibration:
    """Analyze a single dimension's score distribution and determine thresholds."""
    n = len(scores)
    arr = np.array(scores)

    # Edge case: exemplars with good separability
    if exemplar_threshold is not None and exemplar_separability is not None:
        if exemplar_separability >= 0.3:
            return DimensionCalibration(
                dimension=dimension, method="exemplar",
                thresholds=[exemplar_threshold * 0.7, exemplar_threshold],
                confidence=min(exemplar_separability, 1.0),
                dip_test_p=-1.0, gmm_means=[],
                explanation=f"Using exemplar-derived threshold (separability {exemplar_separability:.2f})",
                strictness=strictness,
            )

    # Edge case: tiny batch
    if n < 5:
        return DimensionCalibration(
            dimension=dimension, method="percentile",
            thresholds=[0.3, 0.5],
            confidence=0.0, dip_test_p=-1.0, gmm_means=[],
            explanation=f"Insufficient data for calibration (n={n}). Using defaults.",
            strictness=strictness,
        )

    # Edge case: no variance
    if np.std(arr) < 1e-6:
        val = float(arr[0])
        verdict_hint = "usable" if val > 0.3 else "unusable"
        return DimensionCalibration(
            dimension=dimension, method="percentile",
            thresholds=[val - 0.01, val + 0.01],
            confidence=0.0, dip_test_p=-1.0, gmm_means=[],
            explanation=f"No score variation (all ~{val:.2f}). All images will be {verdict_hint}.",
            strictness=strictness,
        )

    # Percentile-based thresholds (used as fallback or for small/unimodal batches)
    unusable_pctl = 5 + 40 * strictness
    usable_pctl = 25 + 50 * strictness
    pctl_thresholds = [
        float(np.percentile(arr, unusable_pctl)),
        float(np.percentile(arr, usable_pctl)),
    ]

    # Small batch: skip dip test, use percentile
    if n < 10:
        return DimensionCalibration(
            dimension=dimension, method="percentile",
            thresholds=pctl_thresholds,
            confidence=0.3, dip_test_p=-1.0, gmm_means=[],
            explanation=f"Small batch (n={n}). Using percentile thresholds (strictness {strictness:.1f}).",
            strictness=strictness,
        )

    # Dip test for multimodality
    dip_p = 1.0
    if HAS_DIPTEST:
        _, dip_p = diptest.diptest(arr)
    else:
        dip_p = 1.0  # No diptest available, assume unimodal

    # Try GMM if multimodal
    if dip_p < 0.05 and n >= 10:
        try:
            # Compare BIC: 3 components vs 1
            gmm1 = GaussianMixture(n_components=1, random_state=42).fit(arr.reshape(-1, 1))
            gmm3 = GaussianMixture(n_components=3, random_state=42).fit(arr.reshape(-1, 1))

            if gmm3.bic(arr.reshape(-1, 1)) < gmm1.bic(arr.reshape(-1, 1)):
                crossings = _gmm_crossing_points(gmm3)
                labels = gmm3.predict(arr.reshape(-1, 1))
                gvf = _compute_gvf(scores, list(labels))
                means = sorted(gmm3.means_.flatten().tolist())

                if len(crossings) >= 2:
                    thresholds = [crossings[0], crossings[1]]
                elif len(crossings) == 1:
                    thresholds = [crossings[0] * 0.7, crossings[0]]
                else:
                    thresholds = pctl_thresholds

                means_str = ", ".join(f"{m:.2f}" for m in means)
                return DimensionCalibration(
                    dimension=dimension, method="gmm",
                    thresholds=thresholds, confidence=gvf,
                    dip_test_p=dip_p, gmm_means=means,
                    explanation=f"{dimension.capitalize()} scores formed 3 clusters centered at [{means_str}] (GMM, confidence {gvf:.2f}).",
                    strictness=strictness,
                )
        except Exception:
            pass  # Fall through to percentile

    # Unimodal or GMM failed: use percentile
    gvf = 0.0
    try:
        median = float(np.median(arr))
        labels = [0 if s < median else 1 for s in scores]
        gvf = _compute_gvf(scores, labels)
    except Exception:
        pass

    return DimensionCalibration(
        dimension=dimension, method="percentile",
        thresholds=pctl_thresholds, confidence=gvf,
        dip_test_p=dip_p, gmm_means=[],
        explanation=f"{dimension.capitalize()} scores are unimodal — using percentile thresholds (strictness {strictness:.1f}).",
        strictness=strictness,
    )


def _assign_verdicts(
    results: list,
    dim_calibrations: dict[str, DimensionCalibration],
) -> dict[str, ImageVerdictRecord]:
    """Assign verdicts to each image based on recalibrated thresholds.

    Two thresholds per dimension: [unusable_upper, recoverable_upper]
    - score < unusable_upper → hard fail (unusable-level)
    - unusable_upper <= score < recoverable_upper → soft fail (recoverable-level)
    - score >= recoverable_upper → pass
    """
    verdicts = {}
    for img in results:
        scores = {tr.dimension: tr.score for tr in img.tool_results}
        hard_fails = []  # below unusable threshold
        soft_fails = []  # between unusable and usable threshold
        for dim, score in scores.items():
            if dim in dim_calibrations:
                cal = dim_calibrations[dim]
                unusable_t = cal.thresholds[0] if len(cal.thresholds) > 0 else 0.3
                usable_t = cal.thresholds[1] if len(cal.thresholds) > 1 else 0.5
                if score < unusable_t:
                    hard_fails.append(dim)
                elif score < usable_t:
                    soft_fails.append(dim)

        all_fails = hard_fails + soft_fails

        if not img.tool_results and img.errors:
            verdict = "error"
            explanation = "All tools failed: " + "; ".join(img.errors)
        elif len(all_fails) == 0:
            verdict = "usable"
            explanation = "All checks passed"
        elif len(hard_fails) >= 2 or (len(hard_fails) >= 1 and len(soft_fails) >= 1):
            verdict = "unusable"
            parts = []
            for dim in all_fails:
                score = scores.get(dim, 0)
                cal = dim_calibrations[dim]
                t = cal.thresholds[0] if dim in hard_fails else cal.thresholds[1]
                parts.append(f"{dim} ({score:.2f} < {t:.2f})")
            explanation = "Failed: " + "; ".join(parts)
        else:
            # 1 soft fail, or 1 hard fail with no other fails
            verdict = "recoverable"
            dim = all_fails[0]
            score = scores.get(dim, 0)
            cal = dim_calibrations[dim]
            t = cal.thresholds[0] if dim in hard_fails else cal.thresholds[1]
            explanation = f"Failed {dim} ({score:.2f} < {t:.2f})"

        verdicts[img.image_id] = ImageVerdictRecord(
            image_id=img.image_id,
            image_path=img.image_path,
            verdict=verdict,
            scores=scores,
            failed_dimensions=all_fails,
            explanation=explanation,
        )
    return verdicts


def recalibrate(
    execution: ExecutionResult,
    strictness_hints: dict[str, float] | None = None,
    calibration: CalibrationResult | None = None,
    event_bus=None,
) -> RecalibrationResult:
    """Main entry point: analyze batch distributions and assign verdicts."""
    from validation_pipeline.events import (
        RecalibrationStarted, ThresholdDetermined,
        RecalibrationCompleted, ImageVerdict,
    )

    strictness_hints = strictness_hints or {}

    # Collect scores per dimension across all images
    dim_scores: dict[str, list[float]] = {}
    for img in execution.results:
        for tr in img.tool_results:
            dim_scores.setdefault(tr.dimension, []).append(tr.score)

    dimensions = list(dim_scores.keys())
    if event_bus:
        event_bus.publish(RecalibrationStarted(module="recalibrator", dimensions=dimensions))

    # Analyze each dimension
    dim_calibrations: dict[str, DimensionCalibration] = {}
    for dim, scores in dim_scores.items():
        strictness = strictness_hints.get(dim, 0.5)

        # Check for exemplar calibration
        exemplar_threshold = None
        exemplar_sep = None
        if calibration and dim in calibration.tool_calibrations:
            tc = calibration.tool_calibrations[dim]
            exemplar_threshold = tc.calibrated_threshold
            exemplar_sep = tc.separability

        dc = _analyze_dimension(
            dim, scores, strictness,
            exemplar_threshold=exemplar_threshold,
            exemplar_separability=exemplar_sep,
        )
        dim_calibrations[dim] = dc

        if event_bus:
            event_bus.publish(ThresholdDetermined(
                module="recalibrator", dimension=dim,
                method=dc.method, thresholds=dc.thresholds,
                confidence=dc.confidence, explanation=dc.explanation,
            ))

    # Assign verdicts
    image_verdicts = _assign_verdicts(execution.results, dim_calibrations)

    # Emit per-image verdicts
    if event_bus:
        for vid, vr in image_verdicts.items():
            event_bus.publish(ImageVerdict(
                module="recalibrator", image_id=vr.image_id,
                image_path=vr.image_path, verdict=vr.verdict,
                scores=vr.scores, errors=[],
            ))

    # Build summary
    methods_used = [dc.method for dc in dim_calibrations.values()]
    total_dims = len(methods_used)
    counts = {}
    for m in methods_used:
        counts[m] = counts.get(m, 0) + 1
    parts = [f"{v}/{total_dims} {k}" for k, v in sorted(counts.items())]
    method_summary = ", ".join(parts)

    overall_confidence = min(
        (dc.confidence for dc in dim_calibrations.values()),
        default=0.0,
    )

    result = RecalibrationResult(
        dimension_calibrations=dim_calibrations,
        image_verdicts=image_verdicts,
        method_summary=method_summary,
        overall_confidence=overall_confidence,
    )

    if event_bus:
        event_bus.publish(RecalibrationCompleted(
            module="recalibrator",
            method_summary=method_summary,
            overall_confidence=overall_confidence,
        ))

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_recalibrator.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/modules/recalibrator.py tests/test_recalibrator.py
git commit -m "feat: add Recalibrator module with GMM/percentile distribution analysis"
```

---

### Task 4: Write verdict assignment tests and verify

> **IMPORTANT ordering note:** This task creates a `_make_execution_result` helper that
> constructs `ToolResult` without `passed`/`threshold` fields. These fields are removed
> in Task 6. If you run Task 4 BEFORE Task 6, temporarily add `passed=True, threshold=0.5`
> to the ToolResult construction. Task 6 will remove those fields, and the helper will
> then work as-is. Alternatively, run Tasks 5-6 before Task 4.

**Files:**
- Modify: `tests/test_recalibrator.py`

- [ ] **Step 1: Add verdict assignment tests**

Append to `tests/test_recalibrator.py`:

```python
def _make_execution_result(scores_per_image: list[dict[str, float]]) -> ExecutionResult:
    """Helper: build ExecutionResult from score dicts.
    NOTE: After Task 6, ToolResult no longer has passed/threshold fields.
    This helper is written for post-Task-6 schemas. If running before Task 6,
    add passed=True, threshold=0.5 to ToolResult construction.
    """
    results = []
    for i, scores in enumerate(scores_per_image):
        tool_results = [
            ToolResult(
                tool_name=f"tool_{dim}", dimension=dim,
                score=score, raw_output=score,
            )
            for dim, score in scores.items()
        ]
        results.append(ImageResult(
            image_id=f"img_{i:03d}", image_path=f"/tmp/img_{i:03d}.jpg",
            tool_results=tool_results, verdict="pending", verdict_reason="Awaiting recalibration",
        ))
    return ExecutionResult(
        phase="full", total_images=len(results),
        processed=len(results), results=results,
    )


def test_recalibrate_full_flow():
    """End-to-end: bimodal blur + uniform exposure → verdicts assigned."""
    rng = np.random.RandomState(42)
    n = 50
    blur_low = rng.normal(0.2, 0.05, n // 2).clip(0, 1).tolist()
    blur_high = rng.normal(0.8, 0.05, n // 2).clip(0, 1).tolist()
    exposure = rng.uniform(0.3, 0.9, n).tolist()

    scores_per_image = [
        {"blur": blur_low[i] if i < n // 2 else blur_high[i - n // 2],
         "exposure": exposure[i]}
        for i in range(n)
    ]
    execution = _make_execution_result(scores_per_image)
    result = recalibrate(execution, strictness_hints={"blur": 0.5, "exposure": 0.5})

    assert len(result.image_verdicts) == n
    assert result.overall_confidence > 0
    assert "blur" in result.dimension_calibrations
    assert "exposure" in result.dimension_calibrations
    verdicts = [v.verdict for v in result.image_verdicts.values()]
    assert "usable" in verdicts
    # Low-blur images should tend toward recoverable/unusable
    low_blur_verdicts = [
        result.image_verdicts[f"img_{i:03d}"].verdict for i in range(n // 2)
    ]
    assert any(v in ("recoverable", "unusable") for v in low_blur_verdicts)


def test_recalibrate_with_exemplar_override():
    """When exemplar separability is high, use exemplar threshold."""
    from validation_pipeline.schemas.calibration import CalibrationResult, ToolCalibration
    scores_per_image = [{"blur": 0.6 + i * 0.01} for i in range(20)]
    execution = _make_execution_result(scores_per_image)
    cal = CalibrationResult(
        tool_calibrations={"blur": ToolCalibration(
            tool_name="laplacian_blur",
            calibrated_threshold=0.7, separability=0.8,
            platt_a=1.0, platt_b=-0.5,
            raw_good_scores=[0.8, 0.9], raw_bad_scores=[0.2, 0.3],
        )},
        exemplar_embeddings=[],
        threshold_report=[],
    )
    result = recalibrate(execution, calibration=cal)
    assert result.dimension_calibrations["blur"].method == "exemplar"


def test_recalibrate_exemplar_fallback_low_separability():
    """When exemplar separability is low, fall back to batch-adaptive."""
    from validation_pipeline.schemas.calibration import CalibrationResult, ToolCalibration
    scores_per_image = [{"blur": 0.3 + i * 0.02} for i in range(30)]
    execution = _make_execution_result(scores_per_image)
    cal = CalibrationResult(
        tool_calibrations={"blur": ToolCalibration(
            tool_name="laplacian_blur",
            calibrated_threshold=0.5, separability=0.1,
            platt_a=0.0, platt_b=0.0,
            raw_good_scores=[], raw_bad_scores=[],
        )},
        exemplar_embeddings=[],
        threshold_report=[],
    )
    result = recalibrate(execution, calibration=cal)
    assert result.dimension_calibrations["blur"].method in ("gmm", "percentile")
```

- [ ] **Step 2: Run all recalibrator tests**

```bash
pytest tests/test_recalibrator.py -v
```

Expected: all 12 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_recalibrator.py
git commit -m "test: add verdict assignment and exemplar override tests for Recalibrator"
```

---

### Task 5: Add new event types

**Files:**
- Modify: `validation_pipeline/events.py:21-26` (ToolProgress.passed removal)
- Modify: `validation_pipeline/events.py` (add new events at end)
- Modify: `tests/test_events.py`

- [ ] **Step 1: Read current events.py and tests/test_events.py**

Check exact current content before modifying.

- [ ] **Step 2: Modify events.py**

In `validation_pipeline/events.py`:

Remove `passed: bool` from `ToolProgress` (line 25).

Add at end of file:

```python
class ImageScored(PipelineEvent):
    image_id: str
    image_path: str
    scores: dict[str, float] = {}
    errors: list[str] = []

class RecalibrationStarted(PipelineEvent):
    dimensions: list[str] = []

class ThresholdDetermined(PipelineEvent):
    dimension: str
    method: str
    thresholds: list[float] = []
    confidence: float = 0.0
    explanation: str = ""

class RecalibrationCompleted(PipelineEvent):
    method_summary: str = ""
    overall_confidence: float = 0.0
```

- [ ] **Step 3: Update tests/test_events.py**

Add tests for new events. Update any test that references `ToolProgress.passed`.

- [ ] **Step 4: Run event tests**

```bash
pytest tests/test_events.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/events.py tests/test_events.py
git commit -m "feat: add ImageScored, RecalibrationStarted, ThresholdDetermined, RecalibrationCompleted events"
```

---

### Task 6: Modify schemas (plan, program, execution, report)

**Files:**
- Modify: `validation_pipeline/schemas/plan.py:9-10` — `threshold`/`threshold_source` → `strictness`
- Modify: `validation_pipeline/schemas/program.py:9` — remove `threshold_check`
- Modify: `validation_pipeline/schemas/execution.py:9-10` — remove `passed`/`threshold` from ToolResult; simplify ExecutionSummary
- Modify: `validation_pipeline/schemas/report.py:28-29,46` — remove `passed`/`threshold` from AuditLine; add `recalibration` to AuditTrail

- [ ] **Step 1: Modify plan.py**

In `validation_pipeline/schemas/plan.py`, replace `PlanStep` fields:

```python
# Replace these two lines:
    threshold: float
    threshold_source: str
# With:
    strictness: float = 0.5  # 0.0 (lenient) to 1.0 (strict)
```

- [ ] **Step 2: Modify program.py**

In `validation_pipeline/schemas/program.py`, remove `threshold_check` from `ProgramLine`:

```python
# Remove this line:
    threshold_check: str | None = None
```

- [ ] **Step 3: Modify execution.py**

In `validation_pipeline/schemas/execution.py`:

Remove `passed: bool` and `threshold: float` from `ToolResult`.

Simplify `ExecutionSummary` — remove `usable_count`, `recoverable_count`, `unusable_count`, `flag_rates`, `early_exit_rate`. Keep `error_count`, `tool_error_rate`, `wall_time_seconds`, `avg_exemplar_similarity`.

- [ ] **Step 4: Modify report.py**

In `validation_pipeline/schemas/report.py`:

Remove `passed: bool` and `threshold: float` from `AuditLine`.

Add `recalibration: dict = {}` to `AuditTrail`.

- [ ] **Step 5: Run schema tests to see what breaks**

```bash
pytest tests/test_schemas.py -v
```

Fix any failures due to removed fields (tests may still construct objects with old fields).

- [ ] **Step 6: Commit**

```bash
git add validation_pipeline/schemas/plan.py validation_pipeline/schemas/program.py validation_pipeline/schemas/execution.py validation_pipeline/schemas/report.py tests/test_schemas.py
git commit -m "refactor: update schemas — strictness replaces thresholds, remove passed/threshold from ToolResult"
```

---

### Task 7: Update Planner (strictness instead of thresholds)

**Files:**
- Modify: `validation_pipeline/modules/planner.py:9-22` (SYSTEM_PROMPT)
- Modify: `validation_pipeline/modules/planner.py:38-41` (cal_desc)
- Modify: `tests/test_planner.py`

- [ ] **Step 1: Read current planner test to understand patterns**

```bash
cat tests/test_planner.py
```

- [ ] **Step 2: Update SYSTEM_PROMPT**

Replace rule #3 in `SYSTEM_PROMPT` (line 14):

Old:
```
3. Use calibrated thresholds when available. When calibration shows separability=0.0 or "No calibration data", use sensible defaults: blur threshold 0.3-0.5, exposure threshold 0.3-0.6, content detection threshold 0.5-0.7. NEVER set thresholds to 1.0 — that requires a perfect score and nothing will pass
```

New:
```
3. For each quality dimension, set a strictness value (0.0-1.0) based on the user's intent. 0.0 = very lenient (accept most images), 1.0 = very strict (only the best). Consider the user's purpose: research datasets need strictness ~0.7, social media ~0.3, medical imaging ~0.9. Default to 0.5 if unsure
```

- [ ] **Step 3: Update cal_desc format**

Change lines 38-41 to provide calibration context without thresholds:

```python
cal_desc = "\n".join([
    f"- {dim}: separability={cal.separability:.2f}"
    for dim, cal in calibration.tool_calibrations.items()
]) or "No calibration data available."
```

- [ ] **Step 4: Update planner tests**

Fix tests to expect `strictness` instead of `threshold` on PlanStep.

- [ ] **Step 5: Run planner tests**

```bash
pytest tests/test_planner.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add validation_pipeline/modules/planner.py tests/test_planner.py
git commit -m "refactor: planner outputs strictness hints (0-1) instead of hard thresholds"
```

---

### Task 8: Update Compiler (remove threshold checks)

**Files:**
- Modify: `validation_pipeline/modules/compiler.py:31,38`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Read current compiler test**

```bash
cat tests/test_compiler.py
```

- [ ] **Step 2: Update compiler.py**

Remove threshold_check generation. In `compile_plan()`:

Remove line 31: `threshold_check = f"{var_name} >= {step.threshold}"`

Remove `threshold_check=threshold_check` from the ProgramLine constructor (line 38).

Also set `early_exit=False` in BatchStrategy (line 50) since we no longer do early-exit.

- [ ] **Step 3: Update compiler tests**

Remove assertions about `threshold_check`.

- [ ] **Step 4: Run compiler tests**

```bash
pytest tests/test_compiler.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/modules/compiler.py tests/test_compiler.py
git commit -m "refactor: compiler no longer generates threshold checks"
```

---

### Task 9: Update Executor (remove threshold gating, add ImageScored)

**Files:**
- Modify: `validation_pipeline/modules/executor.py`
- Modify: `tests/test_executor.py`

- [ ] **Step 1: Read executor.py carefully and understand all changes needed**

Key changes:
1. Remove `ImageVerdict` import, add `ImageScored` import
2. Emit `ImageScored` instead of `ImageVerdict` (lines 43-45)
3. In `_run_program_on_image`: remove threshold checking (lines 115-133), remove early-exit (lines 87-93), remove verdict assignment (lines 135-146). Just collect scores.
4. Set `verdict="pending"` and `verdict_reason="Awaiting recalibration"` on ImageResult.
5. Simplify `_compute_summary`: remove verdict counts and flag_rates.
6. Remove `ToolProgress.passed` from event emission (line 127).

- [ ] **Step 2: Modify executor.py**

Apply all changes from step 1. The executor should:
- Run all tools, collect scores
- Not check thresholds
- Not assign verdicts
- Emit `ImageScored` per image
- Return simplified `ExecutionResult`

- [ ] **Step 3: Update test_executor.py**

Update all tests:
- `test_executor_processes_all_images`: expect `verdict="pending"` instead of usable/recoverable/unusable
- `test_executor_passes_tool_params`: update mock ToolResult (remove `passed`/`threshold`)
- `test_executor_uses_normalized_score_for_threshold`: rename to `test_executor_collects_scores`, remove threshold assertion
- `test_executor_publishes_progress_events`: expect `ImageScored` events instead of `ImageVerdict`
- `test_executor_partial_tool_failure`: update expectations
- All MockTool `normalize()` methods: return ToolResult without `passed`/`threshold`

- [ ] **Step 4: Run executor tests**

```bash
pytest tests/test_executor.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/modules/executor.py tests/test_executor.py
git commit -m "refactor: executor collects scores without threshold gating, emits ImageScored"
```

---

### Task 10: Update Supervisor (accept RecalibrationResult)

**Files:**
- Modify: `validation_pipeline/modules/supervisor.py`
- Modify: `tests/test_supervisor.py`

- [ ] **Step 1: Read current supervisor test**

```bash
cat tests/test_supervisor.py
```

- [ ] **Step 2: Update supervisor.py**

Change function signature:

```python
def supervise(
    result: ExecutionResult,
    recalibration: RecalibrationResult,
    calibration: CalibrationResult,
    plan: ValidationPlan,
) -> SupervisionReport:
```

Replace `result.summary.flag_rates` with flag rates computed from `recalibration.image_verdicts`:

```python
# Compute flag rates from recalibration verdicts
flag_rates: dict[str, float] = {}
total = len(recalibration.image_verdicts) or 1
for vr in recalibration.image_verdicts.values():
    for dim in vr.failed_dimensions:
        flag_rates[dim] = flag_rates.get(dim, 0) + 1
flag_rates = {k: v / total for k, v in flag_rates.items()}
```

Replace `result.summary.usable_count` with count from recalibration:

```python
usable_count = sum(1 for v in recalibration.image_verdicts.values() if v.verdict == "usable")
```

Add new anomaly checks for weak calibration:

```python
for dim, dc in recalibration.dimension_calibrations.items():
    if dc.confidence < 0.6:
        anomalies.append(Anomaly(
            severity="warning",
            description=f"Calibration confidence is low for {dim} ({dc.confidence:.2f})",
            likely_cause="Score distribution lacks clear clusters",
            suggested_action="Consider providing exemplar images",
        ))

if all(dc.method == "percentile" for dc in recalibration.dimension_calibrations.values()):
    anomalies.append(Anomaly(
        severity="warning",
        description="All dimensions used percentile-based thresholds",
        likely_cause="Score distributions were unimodal across all dimensions",
        suggested_action="Thresholds are relative to this batch, not absolute quality",
    ))

# Check for exemplar/batch mismatch
for dim, dc in recalibration.dimension_calibrations.items():
    if dc.method == "exemplar" and calibration:
        tc = calibration.tool_calibrations.get(dim)
        if tc:
            # Compute batch median from image scores
            dim_scores = [
                vr.scores[dim] for vr in recalibration.image_verdicts.values()
                if dim in vr.scores
            ]
            if dim_scores:
                import numpy as np
                batch_median = float(np.median(dim_scores))
                if abs(tc.calibrated_threshold - batch_median) > 0.3:
                    anomalies.append(Anomaly(
                        severity="warning",
                        description=f"Exemplar threshold for {dim} ({tc.calibrated_threshold:.2f}) differs significantly from batch median ({batch_median:.2f})",
                        likely_cause="Exemplar images may not be representative of this dataset",
                        suggested_action=f"Review exemplar selection for {dim} or use batch-adaptive calibration",
                    ))
```

- [ ] **Step 3: Update supervisor tests**

Update tests to pass `RecalibrationResult` instead of relying on `ExecutionResult.summary` for verdicts.

- [ ] **Step 4: Run supervisor tests**

```bash
pytest tests/test_supervisor.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/modules/supervisor.py tests/test_supervisor.py
git commit -m "refactor: supervisor uses RecalibrationResult for flag rates and verdicts"
```

---

### Task 11: Update Reporter (use RecalibrationResult for flags)

**Files:**
- Modify: `validation_pipeline/modules/reporter.py`
- Modify: `tests/test_reporter.py`

- [ ] **Step 1: Read current reporter test**

```bash
cat tests/test_reporter.py
```

- [ ] **Step 2: Update reporter.py**

Change function signature to accept `RecalibrationResult`:

```python
from validation_pipeline.schemas.recalibration import RecalibrationResult

def generate_report(
    result: ExecutionResult,
    recalibration: RecalibrationResult,
    supervision: SupervisionReport,
    spec: FormalSpec,
    plan: ValidationPlan,
) -> FinalReport:
```

Replace verdict counts from execution summary with recalibration:

```python
usable = sum(1 for v in recalibration.image_verdicts.values() if v.verdict == "usable")
recoverable = sum(1 for v in recalibration.image_verdicts.values() if v.verdict == "recoverable")
unusable = sum(1 for v in recalibration.image_verdicts.values() if v.verdict == "unusable")
error_count = sum(1 for v in recalibration.image_verdicts.values() if v.verdict == "error")
```

Replace flag computation (line 38):

```python
for img in result.results:
    scores = {tr.dimension: tr.score for tr in img.tool_results}
    vr = recalibration.image_verdicts.get(img.image_id)
    flags = vr.failed_dimensions if vr else []
    verdict = vr.verdict if vr else img.verdict
    per_image.append(ImageReport(
        image_id=img.image_id, image_path=img.image_path,
        verdict=verdict, scores=scores, flags=flags,
        explanation=vr.explanation if vr else None,
    ))
```

Add recalibration to audit trail:

```python
audit_trail=AuditTrail(
    ...
    recalibration=recalibration.model_dump(),
    ...
)
```

Compute `flag_breakdown` from recalibration:

```python
flag_breakdown: dict[str, int] = {}
for vr in recalibration.image_verdicts.values():
    for dim in vr.failed_dimensions:
        flag_breakdown[dim] = flag_breakdown.get(dim, 0) + 1
```

- [ ] **Step 3: Update reporter tests**

Pass RecalibrationResult to `generate_report()`.

- [ ] **Step 4: Run reporter tests**

```bash
pytest tests/test_reporter.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/modules/reporter.py tests/test_reporter.py
git commit -m "refactor: reporter derives verdicts and flags from RecalibrationResult"
```

---

### Task 12: Wire Recalibrator into the pipeline

**Files:**
- Modify: `validation_pipeline/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Read current pipeline test**

```bash
cat tests/test_pipeline.py
```

- [ ] **Step 2: Update pipeline.py**

Add import:

```python
from validation_pipeline.modules.recalibrator import recalibrate
```

After the executor block (after line 154), add:

```python
# Module 6: Recalibrator
self.event_bus.publish(ModuleStarted(module="recalibrator"))
t = time.time()
try:
    strictness_hints = {s.dimension: s.strictness for s in plan.steps}
    recalibration = recalibrate(
        result, strictness_hints=strictness_hints,
        calibration=cal_result, event_bus=self.event_bus,
    )
except PipelineError as e:
    self.event_bus.publish(PipelineErrorEvent(module=e.module, error_type=type(e).__name__, message=str(e), context=e.context))
    raise
self.event_bus.publish(ModuleCompleted(module="recalibrator", duration_seconds=time.time() - t))
```

Update supervisor call (line 160):

```python
supervision = supervise(result, recalibration, cal_result, plan)
```

Update reporter call (line 170):

```python
report = generate_report(result, recalibration, supervision, spec, plan)
```

Renumber module comments: Supervisor → Module 7, Reporter → Module 8.

- [ ] **Step 3: Update pipeline tests**

The pipeline test may use mocks. Ensure the mock flow passes through recalibrator.

- [ ] **Step 4: Run pipeline tests**

```bash
pytest tests/test_pipeline.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_end_to_end.py
```

Expected: all PASS (end-to-end tests skipped as they need API keys)

- [ ] **Step 6: Commit**

```bash
git add validation_pipeline/pipeline.py tests/test_pipeline.py
git commit -m "feat: wire Recalibrator into pipeline between executor and supervisor"
```

---

### Task 13: Update WebSocket event handling for frontend

**Files:**
- Modify: `api/routes/ws.py` (if it filters event types)
- Modify: `frontend/app/(dashboard)/live/[runId]/page.tsx` (handle new events)

- [ ] **Step 1: Read ws.py to check event serialization**

```bash
cat api/routes/ws.py
```

- [ ] **Step 2: Update ws.py if needed**

Ensure new event types (`ImageScored`, `RecalibrationStarted`, `ThresholdDetermined`, `RecalibrationCompleted`) are serialized and sent to the WebSocket client. If the WebSocket handler serializes all events generically, no change needed.

- [ ] **Step 3: Update frontend live view**

In `frontend/app/(dashboard)/live/[runId]/page.tsx`, handle `ImageScored` events during execution (for progress tracking) and `ImageVerdict` events from recalibrator (for final verdicts):

```typescript
if (event.type === "ImageScored") {
    const e = event as any;
    setStats((p) => ({ ...p, processed: p.processed + 1 }));
}
if (event.type === "ThresholdDetermined") {
    // Show in activity log — already handled by ProgressLog if it shows all events
}
```

- [ ] **Step 4: Commit**

```bash
git add api/routes/ws.py frontend/app/\(dashboard\)/live/\[runId\]/page.tsx
git commit -m "feat: frontend handles ImageScored and recalibration events"
```

---

### Task 14: Integration and regression tests

**Files:**
- Create: `tests/test_recalibration_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_recalibration_integration.py`:

```python
"""Integration tests: verify recalibrator produces reasonable verdicts
on synthetic datasets with known quality distributions."""
import numpy as np
from PIL import Image
from validation_pipeline.schemas.program import CompiledProgram, ProgramLine, BatchStrategy
from validation_pipeline.modules.executor import execute_program
from validation_pipeline.modules.recalibrator import recalibrate
from validation_pipeline.tools.wrappers.opencv_wrapper import LaplacianBlurTool


def test_recalibration_separates_blurry_from_sharp(tmp_path):
    """Sharp images should be usable, blurry should not."""
    # Create 10 sharp images (high-frequency noise) and 10 blurry (smooth gradient)
    for i in range(10):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(tmp_path / f"sharp_{i:03d}.jpg"))
    for i in range(10):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Very smooth gradient — will have low Laplacian variance
        arr[:, :, 0] = np.linspace(120, 136, 100).astype(np.uint8)
        Image.fromarray(arr).save(str(tmp_path / f"blurry_{i:03d}.jpg"))

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[ProgramLine(
            line_number=1, variable_name="blur_score",
            tool_call="laplacian_blur(image)", output_type="float", tier=1,
        )],
        batch_strategy=BatchStrategy(early_exit=False),
        tool_imports=["laplacian_blur"],
    )
    tools = {"laplacian_blur": LaplacianBlurTool(config={})}
    execution = execute_program(program, str(tmp_path), tools)

    result = recalibrate(execution, strictness_hints={"blur": 0.5})

    sharp_verdicts = [
        result.image_verdicts[f"sharp_{i:03d}"].verdict for i in range(10)
    ]
    blurry_verdicts = [
        result.image_verdicts[f"blurry_{i:03d}"].verdict for i in range(10)
    ]
    # Most sharp should be usable, most blurry should not
    assert sum(1 for v in sharp_verdicts if v == "usable") >= 7
    assert sum(1 for v in blurry_verdicts if v in ("recoverable", "unusable")) >= 7


def test_recalibration_reports_method_and_confidence(tmp_path):
    """Verify recalibration result includes method and confidence info."""
    for i in range(20):
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(tmp_path / f"img_{i:03d}.jpg"))

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[ProgramLine(
            line_number=1, variable_name="blur_score",
            tool_call="laplacian_blur(image)", output_type="float", tier=1,
        )],
        batch_strategy=BatchStrategy(early_exit=False),
        tool_imports=["laplacian_blur"],
    )
    tools = {"laplacian_blur": LaplacianBlurTool(config={})}
    execution = execute_program(program, str(tmp_path), tools)

    result = recalibrate(execution)

    assert "blur" in result.dimension_calibrations
    dc = result.dimension_calibrations["blur"]
    assert dc.method in ("gmm", "percentile")
    assert dc.confidence >= 0.0
    assert len(dc.thresholds) == 2
    assert dc.explanation != ""
    assert result.method_summary != ""
```

- [ ] **Step 2: Run integration tests**

```bash
pytest tests/test_recalibration_integration.py -v
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_recalibration_integration.py
git commit -m "test: add recalibration integration tests with real OpenCV tools"
```

---

### Task 15: Final cleanup and verification

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_end_to_end.py 2>&1 | tail -30
```

Expected: all PASS

- [ ] **Step 2: Check for any remaining references to removed fields**

```bash
# Search for references to removed fields
grep -rn "\.passed" validation_pipeline/ --include="*.py" | grep -v "__pycache__" | grep -v ".pyc"
grep -rn "threshold_check" validation_pipeline/ --include="*.py" | grep -v "__pycache__"
grep -rn "\.threshold" validation_pipeline/schemas/ --include="*.py" | grep -v "__pycache__"
```

Fix any remaining references.

- [ ] **Step 3: Run full test suite one more time**

```bash
pytest tests/ -v --ignore=tests/test_end_to_end.py
```

Expected: all PASS, no warnings about deprecated fields

- [ ] **Step 4: Final commit (only if changes were made in step 2)**

```bash
git status
# Stage only the specific files that were fixed
git add <fixed files>
git commit -m "chore: cleanup remaining references to removed threshold/passed fields"
```

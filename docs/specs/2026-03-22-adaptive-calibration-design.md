# Design Spec: Post-Execution Adaptive Calibration

> Date: 2026-03-22
> Status: Approved
> Research: [calibration-research.md](../research/calibration-research.md)
> Phase: 1 of 4 (batch-adaptive calibration, no new ML dependencies)

## Problem Statement

The pipeline's calibrator uses Platt scaling on good/bad exemplar images. When no exemplars are provided (the common case), all thresholds default to 0.5 with hardcoded normalization curves:

- **Blur**: Fixed sigmoid centered at Laplacian variance 500. Datasets with variance ~100 (phone photos) or ~3000 (DSLR) produce scores clustered at one extreme.
- **Exposure**: Composite formula with hardcoded weights. Reasonable but not adaptive.
- **Content detection**: API confidence passed through directly. Threshold 0.5 is arbitrary — some object classes consistently score 0.3 even when correctly detected.
- **PixelStats**: `raw/80`, capped at 1.0. Never calibrated.

The result is that verdicts (usable/recoverable/unusable) are unreliable for most runs. The system has no self-awareness about its own confidence.

## Architecture Change

### Current flow

```
Calibrator (exemplars) → Planner → Compiler → Executor → Supervisor → Reporter
                                                ↑
                                    thresholds baked in
```

### New flow

```
Planner → Compiler → Executor (no thresholds, collect all scores)
                         ↓
                    Recalibrator (analyzes batch distribution, assigns verdicts)
                         ↓
                    Supervisor → Reporter
```

### What changes

- **Executor** runs ALL tools on ALL images — no early-exit, no threshold gating during execution. Returns raw scores only.
- **Recalibrator** (new module) receives the full score matrix (images x dimensions), determines thresholds from batch distribution, assigns verdicts.
- **Planner** no longer sets hard thresholds. Instead sets `strictness` hints per dimension (0.0 = lenient, 1.0 = strict).
- **Calibrator** (existing Platt scaling) is kept as optional input. When exemplars exist, Recalibrator uses Platt thresholds as a trusted method rather than computing from batch distribution.

### What stays the same

- All tool wrappers, raw scoring, and normalization logic
- Spec generator
- Event system, persistence, frontend
- Supervisor (receives recalibrated verdicts instead of executor verdicts)
- Reporter (enhanced with threshold explanations)

### Trade-off: Early-Exit

Removing early-exit means every image gets scored by every tool, including API calls. For a 50-image batch with 2 API tools, that's ~100 API calls that early-exit might have skipped. The calibration quality improvement justifies this cost. Smart early-exit can be reintroduced later once reliable threshold estimates exist from prior runs.

---

## Recalibrator Module

### Input

- `ExecutionResult` — per-image scores across all dimensions
- `strictness_hints: dict[str, float]` — from planner, per dimension (0.0–1.0)
- `calibration: CalibrationResult | None` — from existing calibrator (if exemplars provided)

### Processing — Three Steps

#### Step 1: Score Distribution Analysis (per dimension)

```
For each dimension (e.g., "blur"):
  1. Collect all scores: [0.82, 0.15, 0.91, 0.23, 0.88, ...]
  2. Run Hartigan dip test → p-value
  3. Branch:
     - If multimodal (p < 0.05):
         Fit 3-component GMM (sklearn GaussianMixture)
         Thresholds = crossing points between adjacent Gaussians
         Validate with BIC (confirm 3 components beats 1)
         method = "gmm"
     - If unimodal (p >= 0.05):
         Use percentile-based thresholds (adjusted by strictness)
         method = "percentile"
  4. Compute GVF (Goodness of Variance Fit)
     If GVF < 0.6, flag as weak calibration in report
```

#### Step 2: Intent-Aware Adjustment

The planner sets `strictness` per dimension (0.0–1.0) based on user intent. This adjusts percentile boundaries for the percentile fallback method:

| Strictness | Unusable (below) | Recoverable | Usable (above) |
|-----------|-------------------|-------------|----------------|
| 0.0 (lenient) | 5th percentile | 5th–25th | 25th |
| 0.2 | 10th | 10th–35th | 35th |
| 0.5 (default) | 20th | 20th–50th | 50th |
| 0.8 (strict) | 35th | 35th–65th | 65th |
| 1.0 (very strict) | 45th | 45th–75th | 75th |

For intermediate strictness values, use continuous formulas:
```
unusable_percentile = 5 + 40 * strictness
usable_percentile   = 25 + 50 * strictness
```
These produce the table above at 0.0, 0.5, and 1.0, and interpolate smoothly between.

GMM thresholds are data-driven and NOT shifted by strictness — the data speaks for itself when clear clusters exist.

#### Step 3: Verdict Assignment

For each image, apply per-dimension thresholds:

- ALL dimensions pass → **usable**
- Exactly 1 dimension fails → **recoverable**
- 2+ dimensions fail → **unusable**
- Tool errors prevented scoring on any dimension → **error**

This verdict logic is unchanged from current behavior. The improvement is that thresholds are now data-informed.

### Output

`RecalibrationResult` containing per-dimension calibrations and per-image verdicts with explanations.

---

## Data Structures

### New schemas (`validation_pipeline/schemas/recalibration.py`)

```python
class DimensionCalibration(BaseModel):
    dimension: str              # "blur", "exposure", "content"
    method: str                 # "gmm" | "percentile" | "exemplar"
    thresholds: list[float]     # [unusable_upper, recoverable_upper]
    confidence: float           # GVF score (0-1), higher = better separation
    dip_test_p: float           # Hartigan p-value (< 0.05 = multimodal)
    gmm_means: list[float]      # cluster centers if GMM, empty otherwise
    explanation: str            # "Blur scores formed 3 clusters centered at 0.2, 0.5, 0.9"
    strictness: float           # planner's hint (0-1)

class ImageVerdictRecord(BaseModel):
    """Per-image verdict assigned by recalibrator. Separate from the ImageVerdict event."""
    image_id: str
    image_path: str
    verdict: str                # "usable" | "recoverable" | "unusable" | "error"
    scores: dict[str, float]    # dimension → score
    failed_dimensions: list[str]  # dimensions that fell below threshold
    explanation: str            # "Failed blur (0.23 < 0.52)"

class RecalibrationResult(BaseModel):
    dimension_calibrations: dict[str, DimensionCalibration]
    image_verdicts: dict[str, ImageVerdictRecord]  # image_id → verdict record
    method_summary: str         # "2/3 dimensions used GMM, 1 used percentile"
    overall_confidence: float   # min confidence across dimensions
```

### Modified schemas

**`validation_pipeline/schemas/plan.py` — PlanStep:**
- `PlanStep.threshold: float` → replaced by `PlanStep.strictness: float` (0.0–1.0)
- `PlanStep.threshold_source: str` → removed (method is now determined by Recalibrator, not planner)

**`validation_pipeline/schemas/program.py` — ProgramLine:**
- `ProgramLine.threshold_check: str | None` → removed (no threshold gating during execution)

**`validation_pipeline/schemas/execution.py` — ToolResult & ExecutionSummary:**
- `ToolResult.passed: bool` → removed (verdicts assigned by Recalibrator, not executor)
- `ToolResult.threshold: float` → removed
- `ExecutionSummary` simplified: remove `usable_count`, `recoverable_count`, `unusable_count`, `flag_rates`, `early_exit_rate`. Keep `wall_time_seconds`, `tool_error_rate`, `error_count`.

**`validation_pipeline/schemas/report.py` — AuditTrail & AuditLine:**
- `AuditTrail.recalibration: dict` → **added** (serialized RecalibrationResult, kept as dict for consistency with existing `calibration: dict` field)
- `AuditLine.passed: bool` → removed
- `AuditLine.threshold: float` → removed (thresholds now live in RecalibrationResult)

**`validation_pipeline/events.py` — ToolProgress:**
- `ToolProgress.passed: bool` → removed (executor no longer determines pass/fail)

---

## Module Integration

### Planner changes

System prompt updated. Instead of:
> "Use calibrated thresholds when available. When calibration shows separability=0.0, use sensible defaults: blur threshold 0.3-0.5..."

New prompt:
> "For each quality dimension, set a strictness value (0.0-1.0) based on the user's intent. 0.0 = very lenient (accept most images), 1.0 = very strict (only the best). Consider the user's purpose: research datasets need strictness ~0.7, social media ~0.3, medical imaging ~0.9."

### Compiler changes

Execution lines no longer contain threshold checks. Each line is just: run this tool on this dimension, collect the score.

### Executor changes

Simplified. For each image:
1. Load image
2. Run all tools in tier order (no early-exit by tier)
3. Collect raw scores and normalized scores
4. No pass/fail determination — just scores
5. Emit `ImageScored` event (replaces `ImageVerdict` during execution)
6. **Remove** `ImageVerdict` emission from executor — this event is now emitted by Recalibrator

`_compute_summary()` simplified: only computes `wall_time_seconds`, `tool_error_rate`, and `error_count`. All verdict-based stats (`usable_count`, `recoverable_count`, `unusable_count`, `flag_rates`, `early_exit_rate`) are removed — these are now computed by the Recalibrator and included in `RecalibrationResult`.

Verdicts are emitted by the Recalibrator after all images are processed.

### Recalibrator (new module, position 6)

Runs after executor, before supervisor. Receives all scores, analyzes distributions, assigns verdicts.

Pipeline order becomes:
```
0. Dataset Resolver (conditional)
1. Spec Generator
2. Calibrator (optional, if exemplars provided)
3. Planner
4. Compiler
5. Executor
6. Recalibrator  ← NEW
7. Supervisor
8. Reporter
```

### Supervisor changes

**Input change**: Supervisor receives `RecalibrationResult` (for verdict-based checks) in addition to `ExecutionResult` (for tool error rates). Its function signature becomes: `supervise(execution: ExecutionResult, recalibration: RecalibrationResult, ...)`.

- `flag_rates` are now computed from `RecalibrationResult.image_verdicts` and `RecalibrationResult.dimension_calibrations`, not from `ExecutionResult.summary`
- `tool_error_rate` still comes from `ExecutionResult.summary`

New anomaly types:
- `weak_calibration` (warning): GVF < 0.6 on any dimension — "Calibration confidence is low for {dimension}. Consider providing exemplar images."
- `all_unimodal` (warning): No dimension had multimodal distribution — "Score distributions were unimodal; thresholds are percentile-based rather than data-driven."
- `exemplar_batch_mismatch` (warning): Exemplar-derived threshold differs from batch median by > 0.3 — "Exemplar threshold for {dimension} ({exemplar_t}) differs significantly from batch distribution (median {batch_median})."

### Reporter changes

**Input change**: Reporter receives `RecalibrationResult` in addition to existing inputs. Per-image flags are now derived from `ImageVerdictRecord.failed_dimensions` instead of `ToolResult.passed`.

Replace:
```python
flags = [tr.dimension for tr in img.tool_results if not tr.passed]
```
With:
```python
verdict_record = recalibration.image_verdicts[img.image_id]
flags = verdict_record.failed_dimensions
```

`FinalReport` includes:
- Threshold explanation per dimension (human-readable)
- Method used per dimension (gmm/percentile/exemplar)
- Confidence signal per dimension
- Visual-ready data: score histograms with threshold lines (for frontend rendering)

### Event flow (new and modified events)

**New events:**
```python
class ImageScored(PipelineEvent):
    """Emitted by executor (replaces ImageVerdict during execution)."""
    image_id: str
    image_path: str
    scores: dict[str, float]    # dimension → normalized score
    errors: list[str] = []

class RecalibrationStarted(PipelineEvent):
    """Emitted when recalibrator begins analysis."""
    dimensions: list[str]

class ThresholdDetermined(PipelineEvent):
    """Emitted per dimension as thresholds are computed."""
    dimension: str
    method: str                 # "gmm" | "percentile" | "exemplar"
    thresholds: list[float]
    confidence: float
    explanation: str

class RecalibrationCompleted(PipelineEvent):
    """Emitted when all verdicts are assigned."""
    method_summary: str
    overall_confidence: float
```

**Moved events:**
- `ImageVerdict` — now emitted by Recalibrator (after all images scored), not by Executor
- `ToolProgress.passed` field — removed

These stream to frontend via WebSocket. Activity log shows:
> "Analyzing blur distribution... Found 3 natural clusters (GMM, confidence 0.87)"
> "Exposure scores are unimodal — using percentile thresholds (strictness 0.5)"

---

## Backward Compatibility with Exemplars

When exemplars ARE provided:
1. Existing `Calibrator` runs Platt scaling as before
2. Produces `CalibrationResult` with per-dimension thresholds and separability
3. Recalibrator receives this as input
4. For each dimension:
   - If separability >= 0.3: use Platt threshold directly (`method = "exemplar"`)
   - If separability < 0.3: Platt data is unreliable, fall back to batch-adaptive (GMM/percentile)
5. Report explains: "Using exemplar-derived threshold for blur (separability 0.72). Falling back to batch-adaptive for content (exemplar separability 0.15)."

---

## Edge Cases

| Scenario | Batch size | Distribution | Approach |
|----------|-----------|-------------|----------|
| Very small batch | 5–9 | Can't fit GMM reliably | Skip dip test, percentile only. Flag "low confidence — small batch" |
| Tiny batch | 1–4 | Stats meaningless | Fall back to hardcoded defaults (current behavior). Flag "insufficient data for calibration" |
| All scores identical | Any | No variance | All images get same verdict (usable if score > 0.3, else unusable). Flag "no score variation — check tool" |
| One dimension all errors | Any | Missing data | Skip dimension for verdict. Verdict based on remaining dimensions only |
| Heavily skewed (95% at 0.9) | Any | Most images high quality | GMM/percentile will reflect this — most images usable, which is correct for a high-quality dataset |
| Exemplars disagree with batch | Any | Platt says 0.7, batch median 0.4 | Use exemplar thresholds but warn in report |

### Out of scope (YAGNI)

- Multi-modal with 4+ clusters — cap at 3 GMM components (matches 3-verdict system)
- Per-image confidence scores — save for Phase 3 (conformal prediction)
- Cross-dimension correlations (e.g., "blurry AND dark") — each dimension is independent
- Online/streaming recalibration — batch-only for now

---

## Dependencies

| Package | Purpose | Size | Status |
|---------|---------|------|--------|
| `scikit-learn` | `GaussianMixture`, BIC | Already installed | Existing |
| `scipy.stats` | Statistical utilities | Already installed | Existing |
| `numpy` | Array operations | Already installed | Existing |
| `diptest` | Hartigan dip test | ~10KB | **New** (only new dependency). Fallback if unavailable: skip dip test, always use percentile method. |

---

## Testing Strategy

### Unit tests (Recalibrator in isolation)

1. **Bimodal scores**: Two clear Gaussians (means 0.2 and 0.8, std 0.1) → verify GMM finds crossing point near 0.5
2. **Trimodal scores**: Three clusters → verify 2 thresholds placed correctly
3. **Uniform scores**: Flat distribution → verify falls back to percentile
4. **Identical scores**: All 0.5 → verify graceful handling, single verdict
5. **Small batch (N=3)**: Verify falls back to hardcoded defaults
6. **Small batch (N=7)**: Verify uses percentile, not GMM
7. **Strictness adjustment**: Verify percentile boundaries shift with strictness 0.0, 0.5, 1.0
8. **Mixed methods**: 2 dimensions GMM + 1 percentile → verify independent handling
9. **Exemplar override**: Platt threshold with high separability → verify method="exemplar"
10. **Exemplar fallback**: Platt threshold with low separability → verify falls back to batch-adaptive

### Integration test

- Run full pipeline on COCO val subset (20 images, known quality mix)
- Verify blurry images → unusable, sharp images → usable
- Compare verdicts: adaptive vs old hardcoded-0.5. Expect more meaningful separation.

### Regression test

- Save current pipeline output on a reference dataset (verdicts + scores)
- After changes, compare. Verdicts should shift toward correctness, not randomness.
- Track: verdict flip count, average confidence, GVF scores

---

## Future Phases (Not In Scope)

### Phase 2: CLIP-IQA Integration
Add QualiCLIP or CLIP-IQA+ as Tier 1 tool. Produces inherently calibrated [0,1] scores with 0.80+ SRCC against human ratings. Replaces/augments OpenCV quality tools. ~400MB model download.

### Phase 3: Conformal Prediction
Wrap verdict assignment with MAPIE for formal coverage guarantees. Produces prediction sets with confidence levels.

### Phase 4: VLM Pairwise Calibration
Use pairwise VLM comparison on a subsample to establish Elo rankings. GenArena showed 0.86 vs 0.36 Spearman correlation vs absolute scoring. Map Elo to verdict boundaries.

---

## File Changes Summary

| File | Change |
|------|--------|
| `validation_pipeline/modules/recalibrator.py` | **New** — core recalibration module |
| `validation_pipeline/schemas/recalibration.py` | **New** — DimensionCalibration, ImageVerdictRecord, RecalibrationResult |
| `validation_pipeline/pipeline.py` | Modified — insert recalibrator at position 6, pass RecalibrationResult to supervisor/reporter |
| `validation_pipeline/modules/executor.py` | Modified — remove threshold checking, remove `ImageVerdict` emission, add `ImageScored` emission, simplify `_compute_summary()` |
| `validation_pipeline/modules/planner.py` | Modified — strictness hints (0.0–1.0) instead of hard thresholds |
| `validation_pipeline/modules/compiler.py` | Modified — remove threshold checks from ProgramLine generation |
| `validation_pipeline/modules/supervisor.py` | Modified — accept RecalibrationResult, compute flag_rates from it, add new anomaly types |
| `validation_pipeline/modules/reporter.py` | Modified — derive flags from RecalibrationResult.image_verdicts instead of ToolResult.passed |
| `validation_pipeline/schemas/plan.py` | Modified — `PlanStep.threshold`/`threshold_source` → `PlanStep.strictness: float` |
| `validation_pipeline/schemas/program.py` | Modified — remove `ProgramLine.threshold_check` |
| `validation_pipeline/schemas/execution.py` | Modified — remove `ToolResult.passed`/`ToolResult.threshold`, simplify `ExecutionSummary` |
| `validation_pipeline/schemas/report.py` | Modified — add `AuditTrail.recalibration: dict`, remove `AuditLine.passed`/`AuditLine.threshold` |
| `validation_pipeline/events.py` | Modified — add `ImageScored`, `RecalibrationStarted`, `ThresholdDetermined`, `RecalibrationCompleted`; remove `ToolProgress.passed`; `ImageVerdict` stays but moves to recalibrator |
| `tests/test_modules/test_recalibrator.py` | **New** — unit tests (10 test cases) |
| `tests/test_integration/test_recalibration.py` | **New** — integration + regression tests |

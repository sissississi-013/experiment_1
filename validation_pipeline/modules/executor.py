import time
from pathlib import Path
from PIL import Image
from validation_pipeline.schemas.program import CompiledProgram
from validation_pipeline.schemas.execution import (
    ExecutionResult, ImageResult, ToolResult, ExecutionSummary,
)
from validation_pipeline.schemas.calibration import CalibrationResult
from validation_pipeline.tools.base import BaseTool
from validation_pipeline.errors import ToolError
from validation_pipeline.events import ImageProgress, ImageVerdict, ToolProgress


def execute_program(
    program: CompiledProgram,
    dataset_path: str,
    tools: dict[str, BaseTool],
    calibration: CalibrationResult | None = None,
    sample_paths: list[str] | None = None,
    event_bus=None,
) -> ExecutionResult:
    start_time = time.time()

    if sample_paths:
        image_paths = sample_paths
    else:
        dataset_dir = Path(dataset_path)
        image_paths = sorted([
            str(p) for p in dataset_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ])

    results = []
    failed = 0

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
            failed += 1
            results.append(ImageResult(
                image_id=Path(img_path).stem,
                image_path=img_path,
                tool_results=[],
                verdict="error",
                verdict_reason=f"Processing error: {str(e)}",
                errors=[str(e)],
            ))

    summary = _compute_summary(results, time.time() - start_time)

    return ExecutionResult(
        phase="full",
        total_images=len(image_paths),
        processed=len(image_paths) - failed,
        failed_to_process=failed,
        results=results,
        summary=summary,
    )


def _run_program_on_image(
    program: CompiledProgram,
    image: Image.Image,
    image_path: str,
    tools: dict[str, BaseTool],
    calibration: CalibrationResult | None,
    event_bus=None,
) -> ImageResult:
    tool_results = []
    all_passed = True
    lines_executed = 0
    reasons = []
    errors = []

    current_tier = 0
    tier_failed = False

    for line in program.per_image_lines:
        # Early exit: if a previous tier failed, skip higher tiers
        if program.batch_strategy.early_exit and tier_failed and line.tier > current_tier:
            break

        if line.tier > current_tier:
            current_tier = line.tier
            tier_failed = False

        tool_name = line.tool_call.split("(")[0]
        if tool_name not in tools:
            continue

        tool = tools[tool_name]
        params = line.tool_params or {}
        try:
            raw_output = tool.execute(image, **params)
        except Exception as e:
            errors.append(f"{tool_name}: {e}")
            continue
        lines_executed += 1

        dim_key = line.variable_name.replace("_score", "")
        cal = calibration.tool_calibrations.get(dim_key) if calibration else None
        # Skip Platt scaling if coefficients are zero (no exemplars were provided)
        if cal and cal.platt_a == 0.0 and cal.platt_b == 0.0:
            cal = None
        tr = tool.normalize(raw_output, cal)

        threshold = 0.5
        if line.threshold_check and ">=" in line.threshold_check:
            try:
                threshold = float(line.threshold_check.split(">=")[-1].strip())
            except ValueError:
                pass

        passed = tr.score >= threshold

        tr.passed = passed
        tr.threshold = threshold
        if event_bus:
            event_bus.publish(ToolProgress(module="executor", tool_name=tool_name, image_path=image_path, score=tr.score, passed=passed))
        tool_results.append(tr)

        if not passed:
            all_passed = False
            tier_failed = True
            reasons.append(f"{tr.dimension}: {tr.score:.2f} failed threshold {threshold}")

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

    return ImageResult(
        image_id=Path(image_path).stem,
        image_path=image_path,
        tool_results=tool_results,
        verdict=verdict,
        verdict_reason=reason,
        lines_executed=lines_executed,
        errors=errors,
    )


def _compute_summary(results: list[ImageResult], wall_time: float) -> ExecutionSummary:
    usable = sum(1 for r in results if r.verdict == "usable")
    recoverable = sum(1 for r in results if r.verdict == "recoverable")
    unusable = sum(1 for r in results if r.verdict == "unusable")
    error = sum(1 for r in results if r.verdict == "error")

    flag_rates: dict[str, float] = {}
    for r in results:
        for tr in r.tool_results:
            if not tr.passed:
                flag_rates[tr.dimension] = flag_rates.get(tr.dimension, 0) + 1
    total = len(results) or 1
    flag_rates = {k: v / total for k, v in flag_rates.items()}

    # Populate tool_error_rate from per-image errors
    tool_errors: dict[str, int] = {}
    for r in results:
        for err_msg in r.errors:
            tool_name = err_msg.split(":")[0].strip() if ":" in err_msg else "unknown"
            tool_errors[tool_name] = tool_errors.get(tool_name, 0) + 1
    tool_error_rate = {k: v / total for k, v in tool_errors.items()}

    return ExecutionSummary(
        usable_count=usable,
        recoverable_count=recoverable,
        unusable_count=unusable,
        error_count=error,
        flag_rates=flag_rates,
        tool_error_rate=tool_error_rate,
        wall_time_seconds=wall_time,
    )

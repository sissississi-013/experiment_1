from validation_pipeline.schemas.execution import ExecutionResult, ImageResult, ToolResult, ExecutionSummary
from validation_pipeline.schemas.plan import ValidationPlan, PlanStep, SamplingStrategy, CostEstimate
from validation_pipeline.schemas.calibration import CalibrationResult
from validation_pipeline.modules.supervisor import supervise


def _make_result(flag_rate_blur=0.15):
    n = 100
    n_flagged = int(n * flag_rate_blur)
    results = []
    for i in range(n):
        flagged = i < n_flagged
        results.append(ImageResult(
            image_id=f"img_{i}", image_path=f"/data/img_{i}.jpg",
            tool_results=[ToolResult(
                tool_name="laplacian_blur", dimension="blur",
                score=0.2 if flagged else 0.8, passed=not flagged, threshold=0.45,
            )],
            verdict="unusable" if flagged else "usable",
            verdict_reason="blur" if flagged else "ok",
        ))
    return ExecutionResult(
        phase="full", total_images=n, processed=n, results=results,
        summary=ExecutionSummary(
            usable_count=n - n_flagged, unusable_count=n_flagged,
            flag_rates={"blur": flag_rate_blur},
        ),
    )


def _make_plan():
    return ValidationPlan(
        plan_id="p", spec_summary="x", sampling_strategy=SamplingStrategy(),
        steps=[PlanStep(
            step_id=1, dimension="blur", tool_name="laplacian_blur",
            threshold=0.45, threshold_source="default",
            hypothesis="expect ~15% flagged", tier=1,
        )],
        combination_logic="ALL_PASS", estimated_cost=CostEstimate(), user_approved=True,
    )


def _make_cal():
    return CalibrationResult(tool_calibrations={}, exemplar_embeddings=[], threshold_report=[])


def test_supervisor_passes_normal_results():
    report = supervise(_make_result(0.15), _make_cal(), _make_plan())
    assert report.status == "passed"


def test_supervisor_warns_on_high_flag_rate():
    report = supervise(_make_result(0.88), _make_cal(), _make_plan())
    assert report.status in ("warnings", "blocked")
    assert len(report.anomalies) > 0


def test_supervisor_blocks_on_zero_usable():
    report = supervise(_make_result(1.0), _make_cal(), _make_plan())
    assert report.status == "blocked"


from validation_pipeline.schemas.execution import ExecutionResult, ExecutionSummary
from validation_pipeline.schemas.calibration import CalibrationResult
from validation_pipeline.schemas.plan import ValidationPlan, SamplingStrategy, CostEstimate
from validation_pipeline.modules.supervisor import supervise

def test_supervisor_detects_high_error_rate():
    result = ExecutionResult(
        phase="full", total_images=10, processed=10,
        summary=ExecutionSummary(
            usable_count=2, recoverable_count=1, unusable_count=2,
            error_count=5,
        ),
    )
    cal = CalibrationResult(tool_calibrations={}, exemplar_embeddings=[], threshold_report=[])
    plan = ValidationPlan(
        plan_id="p1", spec_summary="test", sampling_strategy=SamplingStrategy(),
        steps=[], combination_logic="ALL_PASS", estimated_cost=CostEstimate(), user_approved=True,
    )
    report = supervise(result, cal, plan)
    error_checks = [c for c in report.checks if c.check_name == "image_error_rate"]
    assert len(error_checks) == 1
    assert not error_checks[0].passed

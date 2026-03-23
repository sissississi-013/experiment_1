from validation_pipeline.schemas.execution import ExecutionResult, ImageResult, ToolResult, ExecutionSummary
from validation_pipeline.schemas.plan import ValidationPlan, PlanStep, SamplingStrategy, CostEstimate
from validation_pipeline.schemas.calibration import CalibrationResult
from validation_pipeline.schemas.recalibration import RecalibrationResult, DimensionCalibration, ImageVerdictRecord
from validation_pipeline.modules.supervisor import supervise


def _make_recalibration(n=100, n_flagged=15):
    image_verdicts = {}
    for i in range(n):
        flagged = i < n_flagged
        image_verdicts[f"img_{i}"] = ImageVerdictRecord(
            image_id=f"img_{i}", image_path=f"/data/img_{i}.jpg",
            verdict="unusable" if flagged else "usable",
            scores={"blur": 0.2 if flagged else 0.8},
            failed_dimensions=["blur"] if flagged else [],
            explanation="Failed blur" if flagged else "All checks passed",
        )
    return RecalibrationResult(
        dimension_calibrations={
            "blur": DimensionCalibration(
                dimension="blur", method="percentile",
                thresholds=[0.3, 0.5], confidence=0.7,
                dip_test_p=0.8, explanation="Percentile thresholds",
                strictness=0.5,
            ),
        },
        image_verdicts=image_verdicts,
        method_summary="1/1 percentile",
        overall_confidence=0.7,
    )


def _make_result():
    return ExecutionResult(
        phase="full", total_images=100, processed=100,
        results=[
            ImageResult(
                image_id=f"img_{i}", image_path=f"/data/img_{i}.jpg",
                tool_results=[ToolResult(
                    tool_name="laplacian_blur", dimension="blur",
                    score=0.8,
                )],
                verdict="pending", verdict_reason="pending",
            )
            for i in range(100)
        ],
        summary=ExecutionSummary(),
    )


def _make_plan():
    return ValidationPlan(
        plan_id="p", spec_summary="x", sampling_strategy=SamplingStrategy(),
        steps=[PlanStep(
            step_id=1, dimension="blur", tool_name="laplacian_blur",
            strictness=0.5,
            hypothesis="expect ~15% flagged", tier=1,
        )],
        combination_logic="ALL_PASS", estimated_cost=CostEstimate(), user_approved=True,
    )


def _make_cal():
    return CalibrationResult(tool_calibrations={}, exemplar_embeddings=[], threshold_report=[])


def test_supervisor_passes_normal_results():
    recal = _make_recalibration(n=100, n_flagged=15)
    report = supervise(_make_result(), recal, _make_cal(), _make_plan())
    # With percentile method, we get the "all percentile" warning
    assert report.status in ("passed", "warnings")


def test_supervisor_warns_on_high_flag_rate():
    recal = _make_recalibration(n=100, n_flagged=88)
    report = supervise(_make_result(), recal, _make_cal(), _make_plan())
    assert report.status in ("warnings", "blocked")
    assert len(report.anomalies) > 0


def test_supervisor_blocks_on_zero_usable():
    recal = _make_recalibration(n=100, n_flagged=100)
    report = supervise(_make_result(), recal, _make_cal(), _make_plan())
    assert report.status == "blocked"


def test_supervisor_detects_high_error_rate():
    result = ExecutionResult(
        phase="full", total_images=10, processed=10,
        summary=ExecutionSummary(error_count=5),
    )
    recal = RecalibrationResult(
        dimension_calibrations={},
        image_verdicts={},
        method_summary="",
        overall_confidence=0.0,
    )
    cal = CalibrationResult(tool_calibrations={}, exemplar_embeddings=[], threshold_report=[])
    plan = ValidationPlan(
        plan_id="p1", spec_summary="test", sampling_strategy=SamplingStrategy(),
        steps=[], combination_logic="ALL_PASS", estimated_cost=CostEstimate(), user_approved=True,
    )
    report = supervise(result, recal, cal, plan)
    error_checks = [c for c in report.checks if c.check_name == "image_error_rate"]
    assert len(error_checks) == 1
    assert not error_checks[0].passed


def test_supervisor_warns_low_confidence():
    recal = _make_recalibration(n=100, n_flagged=15)
    # Set low confidence
    recal.dimension_calibrations["blur"].confidence = 0.3
    report = supervise(_make_result(), recal, _make_cal(), _make_plan())
    low_conf = [a for a in report.anomalies if "confidence is low" in a.description]
    assert len(low_conf) == 1


def test_supervisor_warns_all_percentile():
    recal = _make_recalibration(n=100, n_flagged=15)
    report = supervise(_make_result(), recal, _make_cal(), _make_plan())
    pctl_warnings = [a for a in report.anomalies if "percentile-based" in a.description]
    assert len(pctl_warnings) == 1

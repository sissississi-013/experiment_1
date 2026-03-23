import pytest
from validation_pipeline.schemas.execution import ExecutionResult, ImageResult, ToolResult, ExecutionSummary
from validation_pipeline.schemas.recalibration import RecalibrationResult, DimensionCalibration, ImageVerdictRecord
from validation_pipeline.schemas.supervision import SupervisionReport
from validation_pipeline.schemas.spec import FormalSpec, QualityCriterion, QuantityTarget, OutputFormat
from validation_pipeline.schemas.plan import ValidationPlan, SamplingStrategy, CostEstimate
from validation_pipeline.schemas.report import FinalReport
from validation_pipeline.modules.reporter import generate_report


def test_report_has_curation_score():
    result = ExecutionResult(
        phase="full", total_images=10, processed=10,
        results=[
            ImageResult(
                image_id=f"img_{i}", image_path=f"/img_{i}.jpg",
                tool_results=[ToolResult(
                    tool_name="blur", dimension="blur", score=0.8,
                )],
                verdict="pending", verdict_reason="pending",
            )
            for i in range(8)
        ] + [
            ImageResult(
                image_id=f"img_{i}", image_path=f"/img_{i}.jpg",
                tool_results=[ToolResult(
                    tool_name="blur", dimension="blur", score=0.3,
                )],
                verdict="pending", verdict_reason="pending",
            )
            for i in range(8, 10)
        ],
        summary=ExecutionSummary(),
    )

    # Build recalibration: 8 usable, 2 unusable
    image_verdicts = {}
    for i in range(8):
        image_verdicts[f"img_{i}"] = ImageVerdictRecord(
            image_id=f"img_{i}", image_path=f"/img_{i}.jpg",
            verdict="usable", scores={"blur": 0.8},
            failed_dimensions=[], explanation="All checks passed",
        )
    for i in range(8, 10):
        image_verdicts[f"img_{i}"] = ImageVerdictRecord(
            image_id=f"img_{i}", image_path=f"/img_{i}.jpg",
            verdict="unusable", scores={"blur": 0.3},
            failed_dimensions=["blur"], explanation="Failed blur",
        )
    recalibration = RecalibrationResult(
        dimension_calibrations={
            "blur": DimensionCalibration(
                dimension="blur", method="percentile",
                thresholds=[0.3, 0.5], confidence=0.7,
                dip_test_p=0.8, explanation="Percentile",
                strictness=0.5,
            ),
        },
        image_verdicts=image_verdicts,
        method_summary="1/1 percentile",
        overall_confidence=0.7,
    )

    spec = FormalSpec(
        restated_request="x", assumptions=[], content_criteria=[], success_criteria="x",
        quality_criteria=[QualityCriterion(dimension="blur", description="no blur")],
        quantity_targets=QuantityTarget(), output_format=OutputFormat(),
    )
    plan = ValidationPlan(
        plan_id="p", spec_summary="x", sampling_strategy=SamplingStrategy(),
        steps=[], combination_logic="ALL_PASS", estimated_cost=CostEstimate(), user_approved=True,
    )
    supervision = SupervisionReport(status="passed")

    report = generate_report(result, recalibration, supervision, spec, plan)
    assert isinstance(report, FinalReport)
    assert report.curation_score.overall_score == pytest.approx(0.8, abs=0.01)
    assert report.dataset_stats.usable == 8
    assert report.dataset_stats.flag_breakdown == {"blur": 2}
    assert report.audit_trail.recalibration != {}

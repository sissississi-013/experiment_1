import pytest
from validation_pipeline.schemas.execution import ExecutionResult, ImageResult, ToolResult, ExecutionSummary
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
                    tool_name="blur", dimension="blur", score=0.8, passed=True, threshold=0.5,
                )],
                verdict="usable", verdict_reason="ok",
            )
            for i in range(8)
        ] + [
            ImageResult(
                image_id=f"img_{i}", image_path=f"/img_{i}.jpg",
                tool_results=[ToolResult(
                    tool_name="blur", dimension="blur", score=0.3, passed=False, threshold=0.5,
                )],
                verdict="unusable", verdict_reason="blur",
            )
            for i in range(8, 10)
        ],
        summary=ExecutionSummary(usable_count=8, unusable_count=2, flag_rates={"blur": 0.2}),
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

    report = generate_report(result, supervision, spec, plan)
    assert isinstance(report, FinalReport)
    assert report.curation_score.overall_score == pytest.approx(0.8, abs=0.01)
    assert report.dataset_stats.usable == 8

from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.schemas.spec import (
    FormalSpec, ContentCriterion, QualityCriterion, QuantityTarget, OutputFormat,
)
from validation_pipeline.schemas.plan import ValidationPlan, PlanStep, SamplingStrategy, CostEstimate
from validation_pipeline.schemas.execution import ToolResult


def test_user_input_minimal():
    ui = UserInput(dataset_path="/data/coco", intent="find horses")
    assert ui.dataset_path == "/data/coco"
    assert ui.exemplar_good_paths == []


def test_user_input_with_exemplars():
    ui = UserInput(
        dataset_path="/data/coco",
        intent="find sharp horse images",
        exemplar_good_paths=["/ex/good1.jpg", "/ex/good2.jpg"],
        exemplar_bad_paths=["/ex/bad1.jpg"],
    )
    assert len(ui.exemplar_good_paths) == 2


def test_formal_spec_requires_confirmation():
    spec = FormalSpec(
        restated_request="Find horse images",
        assumptions=["RGB images", "COCO dataset"],
        content_criteria=[ContentCriterion(object_or_scene="horse", must_contain=True, exemplar_based=False)],
        quality_criteria=[QualityCriterion(dimension="blur", description="not blurry")],
        quantity_targets=QuantityTarget(),
        output_format=OutputFormat(),
        success_criteria="Images contain horses and are not blurry",
    )
    assert spec.user_confirmed is False


def test_tool_result_normalized():
    tr = ToolResult(
        tool_name="laplacian_blur",
        dimension="blur",
        score=0.85,
        passed=False,
        threshold=0.45,
    )
    assert tr.passed is False
    assert tr.score > tr.threshold


def test_plan_step_has_hypothesis():
    step = PlanStep(
        step_id=1,
        dimension="blur",
        tool_name="laplacian_blur",
        threshold=0.45,
        threshold_source="exemplar_calibration",
        hypothesis="Laplacian variance will catch 90%+ of blur at threshold 0.45",
    )
    assert step.hypothesis != ""


def test_validation_plan_requires_approval():
    plan = ValidationPlan(
        plan_id="test_001",
        spec_summary="Find sharp horse images",
        sampling_strategy=SamplingStrategy(),
        steps=[],
        estimated_cost=CostEstimate(),
    )
    assert plan.user_approved is False

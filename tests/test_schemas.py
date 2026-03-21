from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.schemas.spec import (
    FormalSpec, ContentCriterion, QualityCriterion, QuantityTarget, OutputFormat,
)
from validation_pipeline.schemas.plan import ValidationPlan, PlanStep, SamplingStrategy, CostEstimate
from validation_pipeline.schemas.execution import ToolResult
from validation_pipeline.schemas.program import ProgramLine


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


def test_plan_step_tool_params():
    step = PlanStep(
        step_id=1, dimension="content", tool_name="roboflow_object_detection",
        threshold=0.5, threshold_source="default",
        hypothesis="Detect horses", tier=2,
        tool_params={"target_label": "horse"},
    )
    assert step.tool_params == {"target_label": "horse"}


def test_plan_step_tool_params_default_none():
    step = PlanStep(
        step_id=1, dimension="blur", tool_name="laplacian_blur",
        threshold=0.5, threshold_source="default",
        hypothesis="Detect blur", tier=1,
    )
    assert step.tool_params is None


def test_program_line_tool_params():
    line = ProgramLine(
        line_number=1, variable_name="content_score",
        tool_call="roboflow_object_detection(image)",
        output_type="float", tier=2,
        tool_params={"target_label": "horse"},
    )
    assert line.tool_params == {"target_label": "horse"}


def test_user_input_optional_dataset_path():
    ui = UserInput(intent="find horses", dataset_description="50 horse images from COCO")
    assert ui.dataset_path is None
    assert ui.dataset_description == "50 horse images from COCO"


def test_user_input_with_dataset_path():
    ui = UserInput(dataset_path="/data/coco", intent="find horses")
    assert ui.dataset_path == "/data/coco"
    assert ui.dataset_description is None

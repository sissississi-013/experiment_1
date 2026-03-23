from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.schemas.spec import (
    FormalSpec, ContentCriterion, QualityCriterion, QuantityTarget, OutputFormat,
)
from validation_pipeline.schemas.plan import ValidationPlan, PlanStep, SamplingStrategy, CostEstimate
from validation_pipeline.schemas.execution import ToolResult, ImageResult, ExecutionSummary
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
    )
    assert tr.score == 0.85


def test_plan_step_has_hypothesis():
    step = PlanStep(
        step_id=1,
        dimension="blur",
        tool_name="laplacian_blur",
        strictness=0.7,
        hypothesis="Laplacian variance will catch 90%+ of blur",
    )
    assert step.hypothesis != ""
    assert step.strictness == 0.7


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
        strictness=0.5,
        hypothesis="Detect horses", tier=2,
        tool_params={"target_label": "horse"},
    )
    assert step.tool_params == {"target_label": "horse"}


def test_plan_step_tool_params_default_none():
    step = PlanStep(
        step_id=1, dimension="blur", tool_name="laplacian_blur",
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

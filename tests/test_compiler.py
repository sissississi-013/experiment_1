import pytest
from validation_pipeline.schemas.plan import ValidationPlan, PlanStep, SamplingStrategy, CostEstimate
from validation_pipeline.schemas.program import CompiledProgram
from validation_pipeline.modules.compiler import compile_plan


def _make_plan(approved=True):
    return ValidationPlan(
        plan_id="plan_001",
        spec_summary="Find sharp horse images",
        sampling_strategy=SamplingStrategy(method="random", sample_rate=0.05),
        steps=[
            PlanStep(step_id=1, dimension="blur", tool_name="laplacian_blur", threshold=250.0,
                     threshold_source="exemplar", hypothesis="catches blur", tier=1, parallel_group=1),
            PlanStep(step_id=2, dimension="exposure", tool_name="histogram_exposure", threshold=0.3,
                     threshold_source="default", hypothesis="catches dark images", tier=1, parallel_group=1),
            PlanStep(step_id=3, dimension="content_match", tool_name="grounding_dino",
                     tool_config={"query": "horse"}, threshold=0.5,
                     threshold_source="default", hypothesis="detects horses", tier=2, parallel_group=2),
        ],
        combination_logic="ALL_PASS",
        estimated_cost=CostEstimate(),
        user_approved=approved,
    )


def test_compile_produces_program():
    program = compile_plan(_make_plan())
    assert isinstance(program, CompiledProgram)
    assert program.source_plan_id == "plan_001"


def test_compile_orders_by_tier():
    program = compile_plan(_make_plan())
    tiers = [line.tier for line in program.per_image_lines]
    assert tiers == sorted(tiers)


def test_compile_has_early_exit():
    program = compile_plan(_make_plan())
    assert program.batch_strategy.early_exit is True


def test_compile_rejects_unapproved_plan():
    with pytest.raises(ValueError, match="not approved"):
        compile_plan(_make_plan(approved=False))


def test_compile_preserves_tool_params():
    plan = ValidationPlan(
        plan_id="p1", spec_summary="test",
        sampling_strategy=SamplingStrategy(),
        steps=[PlanStep(
            step_id=1, dimension="content", tool_name="roboflow_object_detection",
            threshold=0.5, threshold_source="default",
            hypothesis="Detect horse", tier=2,
            tool_params={"target_label": "horse"},
        )],
        combination_logic="ALL_PASS",
        estimated_cost=CostEstimate(),
        user_approved=True,
    )
    program = compile_plan(plan)
    assert program.per_image_lines[0].tool_params == {"target_label": "horse"}


def test_compile_tool_params_none_for_tier1():
    plan = ValidationPlan(
        plan_id="p1", spec_summary="test",
        sampling_strategy=SamplingStrategy(),
        steps=[PlanStep(
            step_id=1, dimension="blur", tool_name="laplacian_blur",
            threshold=100.0, threshold_source="default",
            hypothesis="Detect blur", tier=1,
        )],
        combination_logic="ALL_PASS",
        estimated_cost=CostEstimate(),
        user_approved=True,
    )
    program = compile_plan(plan)
    assert program.per_image_lines[0].tool_params is None

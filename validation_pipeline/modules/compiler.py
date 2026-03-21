import uuid
from validation_pipeline.schemas.plan import ValidationPlan
from validation_pipeline.schemas.program import CompiledProgram, ProgramLine, BatchStrategy
from validation_pipeline.errors import SpecValidationError


def compile_plan(plan: ValidationPlan) -> CompiledProgram:
    if not plan.user_approved:
        raise SpecValidationError(
            "Cannot compile a plan that is not approved by the user",
            module="compiler",
            context={"plan_id": plan.plan_id},
        )

    sorted_steps = sorted(plan.steps, key=lambda s: (s.tier, s.parallel_group, s.step_id))

    lines = []
    tool_imports = []

    for i, step in enumerate(sorted_steps):
        var_name = f"{step.dimension}_score"
        config_parts = []
        for k, v in step.tool_config.items():
            if isinstance(v, str):
                config_parts.append(f'{k}="{v}"')
            else:
                config_parts.append(f"{k}={v}")
        config_str = ", ".join(config_parts)
        tool_call = f"{step.tool_name}(image" + (f", {config_str}" if config_str else "") + ")"

        threshold_check = f"{var_name} >= {step.threshold}"

        lines.append(ProgramLine(
            line_number=i + 1,
            variable_name=var_name,
            tool_call=tool_call,
            output_type="float",
            threshold_check=threshold_check,
            tier=step.tier,
            tool_params=step.tool_params,
        ))

        if step.tool_name not in tool_imports:
            tool_imports.append(step.tool_name)

    return CompiledProgram(
        program_id=str(uuid.uuid4())[:8],
        source_plan_id=plan.plan_id,
        per_image_lines=lines,
        batch_strategy=BatchStrategy(early_exit=True),
        tool_imports=tool_imports,
    )

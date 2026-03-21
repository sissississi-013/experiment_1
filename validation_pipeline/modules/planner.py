import instructor
from openai import OpenAI
from validation_pipeline.schemas.spec import FormalSpec
from validation_pipeline.schemas.calibration import CalibrationResult
from validation_pipeline.schemas.plan import ValidationPlan
from validation_pipeline.config import PipelineConfig


SYSTEM_PROMPT = """You are a validation plan generator. Given a formal specification, calibration results, and available tools, produce a validation plan.

Rules:
1. For each content criterion, select a detection tool (prefer nvidia_grounding_dino for object detection; fallback to roboflow_object_detection)
2. For each quality criterion, select a measurement tool matching the dimension
3. Use calibrated thresholds when available. When calibration shows separability=0.0 or "No calibration data", use sensible defaults: blur threshold 0.3-0.5, exposure threshold 0.3-0.6, content detection threshold 0.5-0.7. NEVER set thresholds to 1.0 — that requires a perfect score and nothing will pass
4. Order steps by tier: Tier 1 (cheap, CPU) first, Tier 2 (API) second, Tier 3 (VLM) last
5. Group independent steps in the same parallel_group
6. Each tool selection MUST include a hypothesis explaining why this tool was chosen and what you expect
7. Set combination_logic to "ALL_PASS" unless the spec suggests otherwise
8. Estimate cost based on tool cost_estimate_ms and expected dataset size
9. For content_detection tools (Tier 2), set tool_params to {"target_label": "<object from content_criteria>"}
10. For semantic_quality tools (Tier 3), set tool_params to {"semantic_question": "<description from criterion>"}
11. For image_quality tools (Tier 1), leave tool_params as null"""


def _call_llm(
    spec: FormalSpec,
    calibration: CalibrationResult,
    tools: list[dict],
    config: PipelineConfig | None = None,
) -> ValidationPlan:
    config = config or PipelineConfig()
    client = instructor.from_openai(OpenAI(api_key=config.openai_api_key))

    tools_desc = "\n".join([
        f"- {t['name']}: task={t.get('task_type','?')}, tier={t.get('tier','?')}, cost={t.get('cost_estimate_ms','?')}ms"
        for t in tools
    ])
    cal_desc = "\n".join([
        f"- {dim}: threshold={cal.calibrated_threshold:.2f}, separability={cal.separability:.2f}"
        for dim, cal in calibration.tool_calibrations.items()
    ]) or "No calibration data available."

    return client.chat.completions.create(
        model=config.llm_model,
        response_model=ValidationPlan,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Spec:\n{spec.model_dump_json(indent=2)}\n\n"
                f"Calibration:\n{cal_desc}\n\n"
                f"Available tools:\n{tools_desc}"
            )},
        ],
        max_retries=config.max_retries,
    )


def generate_plan(
    spec: FormalSpec,
    calibration: CalibrationResult,
    tools: list[dict],
    config: PipelineConfig | None = None,
) -> ValidationPlan:
    plan = _call_llm(spec, calibration, tools, config)
    plan.user_approved = False
    return plan

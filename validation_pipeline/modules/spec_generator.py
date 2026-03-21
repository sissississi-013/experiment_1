import instructor
from openai import OpenAI
from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.schemas.spec import FormalSpec
from validation_pipeline.config import PipelineConfig


SYSTEM_PROMPT = """You are a data validation spec generator. Given a user's dataset and intent, produce a formal specification.

You MUST:
1. Restate the request in clear, unambiguous language
2. List all assumptions (implicit and explicit)
3. Break requirements into content criteria (what must be IN the image) and quality criteria (what quality bar must be met)
4. Set quantity targets if mentioned
5. Define output format preferences

IMPORTANT: For quality_criteria, the dimension field MUST be one of these exact values:
- "blur" (for sharpness, blur, focus issues)
- "exposure" (for brightness, darkness, lighting issues)
- "information_content" (for solid colors, garbage frames, blank images)

Do NOT create duplicate dimensions. "sharp" and "not blurry" both map to dimension="blur". "well-lit" and "not dark" both map to dimension="exposure". Use exactly one quality criterion per dimension.

If exemplar images are provided, mark relevant content criteria as exemplar_based=True."""


def _call_llm(user_input: UserInput, config: PipelineConfig | None = None) -> FormalSpec:
    config = config or PipelineConfig()
    client = instructor.from_openai(OpenAI(api_key=config.openai_api_key))

    exemplar_note = ""
    if user_input.exemplar_good_paths:
        exemplar_note = (
            f"\nUser provided {len(user_input.exemplar_good_paths)} good example images"
            f" and {len(user_input.exemplar_bad_paths)} bad example images."
        )

    return client.chat.completions.create(
        model=config.llm_model,
        response_model=FormalSpec,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Dataset: {user_input.dataset_path}\nIntent: {user_input.intent}{exemplar_note}"},
        ],
        max_retries=config.max_retries,
    )


def generate_spec(user_input: UserInput, config: PipelineConfig | None = None) -> FormalSpec:
    from validation_pipeline.errors import LLMError
    try:
        spec = _call_llm(user_input, config)
    except Exception as e:
        raise LLMError(
            f"Spec generation failed: {e}",
            module="spec_generator",
            context={"intent": user_input.intent},
        ) from e
    spec.user_confirmed = False
    return spec

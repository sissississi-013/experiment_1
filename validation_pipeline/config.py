from pydantic import BaseModel


class PipelineConfig(BaseModel):
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""
    device: str = "cpu"
    tool_configs_dir: str = "validation_pipeline/tools/configs"
    default_sample_rate: float = 0.05
    max_retries: int = 3


def load_config(**overrides) -> PipelineConfig:
    return PipelineConfig(**overrides)

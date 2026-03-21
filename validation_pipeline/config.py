from pydantic import BaseModel


class RetryPolicy(BaseModel):
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0


class PipelineConfig(BaseModel):
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""
    device: str = "cpu"
    tool_configs_dir: str = "validation_pipeline/tools/configs"
    default_sample_rate: float = 0.05
    max_retries: int = 3
    retry_policy: RetryPolicy = RetryPolicy()


def load_config(**overrides) -> PipelineConfig:
    return PipelineConfig(**overrides)

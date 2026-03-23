from pydantic import BaseModel


class PlanStep(BaseModel):
    step_id: int
    dimension: str
    tool_name: str
    tool_config: dict = {}
    strictness: float = 0.5
    hypothesis: str
    fallback_tool: str | None = None
    parallel_group: int = 1
    tier: int = 1
    tool_params: dict | None = None


class SamplingStrategy(BaseModel):
    method: str = "random"
    sample_rate: float = 0.05
    cluster_count: int | None = None


class CostEstimate(BaseModel):
    sample_time_seconds: float = 0.0
    full_run_time_seconds: float = 0.0
    tier1_images: int = 0
    tier2_images: int = 0
    tier3_images: int = 0


class ValidationPlan(BaseModel):
    plan_id: str
    spec_summary: str
    sampling_strategy: SamplingStrategy
    steps: list[PlanStep]
    combination_logic: str = "ALL_PASS"
    estimated_cost: CostEstimate
    user_approved: bool = False

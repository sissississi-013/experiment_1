from typing import Any
from pydantic import BaseModel


class ToolResult(BaseModel):
    tool_name: str
    dimension: str
    score: float
    passed: bool
    threshold: float
    confidence: float = 1.0
    raw_output: Any = None
    explanation: str | None = None
    calibration_method: str = "default"


class ImageResult(BaseModel):
    image_id: str
    image_path: str
    tool_results: list[ToolResult]
    verdict: str
    verdict_reason: str
    lines_executed: int = 0
    exemplar_similarity: float | None = None


class ExecutionSummary(BaseModel):
    usable_count: int = 0
    recoverable_count: int = 0
    unusable_count: int = 0
    flag_rates: dict[str, float] = {}
    avg_exemplar_similarity: float | None = None
    early_exit_rate: float = 0.0
    tool_error_rate: dict[str, float] = {}
    wall_time_seconds: float = 0.0


class ExecutionResult(BaseModel):
    phase: str
    total_images: int
    processed: int = 0
    failed_to_process: int = 0
    results: list[ImageResult] = []
    summary: ExecutionSummary = ExecutionSummary()

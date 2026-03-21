from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

class PipelineEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    module: str

class ModuleStarted(PipelineEvent):
    details: str = ""

class ModuleCompleted(PipelineEvent):
    duration_seconds: float = 0.0
    summary: str = ""

class ImageProgress(PipelineEvent):
    current: int
    total: int
    image_path: str

class ToolProgress(PipelineEvent):
    tool_name: str
    image_path: str
    score: float
    passed: bool

class DatasetResolved(PipelineEvent):
    source: str
    image_count: int
    download_path: str

class SpecGenerated(PipelineEvent):
    spec_summary: str
    quality_criteria: list[str] = []
    content_criteria: list[str] = []

class PlanGenerated(PipelineEvent):
    steps_count: int
    tiers: list[int] = []

class ImageVerdict(PipelineEvent):
    image_id: str
    image_path: str
    verdict: str
    scores: dict[str, float] = {}
    errors: list[str] = []

class PipelineErrorEvent(PipelineEvent):
    error_type: str
    message: str
    context: dict[str, Any] = {}

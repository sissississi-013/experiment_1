from pydantic import BaseModel


class ContentCriterion(BaseModel):
    object_or_scene: str
    must_contain: bool
    exemplar_based: bool


class QualityCriterion(BaseModel):
    dimension: str
    description: str
    threshold_hint: str | None = None


class QuantityTarget(BaseModel):
    min_images: int | None = None
    per_class: bool = False


class OutputFormat(BaseModel):
    format: str = "json"
    include_rejected: bool = True
    include_recoverable: bool = True


class FormalSpec(BaseModel):
    restated_request: str
    assumptions: list[str]
    content_criteria: list[ContentCriterion]
    quality_criteria: list[QualityCriterion]
    quantity_targets: QuantityTarget
    output_format: OutputFormat
    success_criteria: str
    user_confirmed: bool = False

from typing import Any
from pydantic import BaseModel


class DatasetStats(BaseModel):
    total_images: int
    usable: int
    recoverable: int
    unusable: int
    usable_percentage: float
    flag_breakdown: dict[str, int]


class CurationScore(BaseModel):
    overall_score: float
    dimension_scores: dict[str, float]
    confidence: float
    explanation: str


class AuditLine(BaseModel):
    line_number: int
    tool_name: str
    variable: str
    raw_value: Any
    calibrated_value: float
    threshold: float
    passed: bool


class ImageReport(BaseModel):
    image_id: str
    image_path: str
    verdict: str
    scores: dict[str, float]
    flags: list[str]
    recovery_suggestion: str | None = None
    explanation: str | None = None
    audit: list[AuditLine] = []


class AuditTrail(BaseModel):
    spec: dict
    plan: dict
    calibration: dict
    supervision_report: dict
    timestamp: str
    llm_model_used: str
    tool_versions: dict[str, str]


class OutputFiles(BaseModel):
    usable_manifest: str
    full_results_json: str
    recoverable_manifest: str | None = None
    rejected_manifest: str | None = None


class FinalReport(BaseModel):
    report_id: str
    spec_summary: str
    dataset_stats: DatasetStats
    curation_score: CurationScore
    per_image_results: list[ImageReport]
    audit_trail: AuditTrail
    output_files: OutputFiles

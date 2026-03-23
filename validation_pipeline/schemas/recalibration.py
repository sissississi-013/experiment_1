from pydantic import BaseModel


class DimensionCalibration(BaseModel):
    dimension: str
    method: str  # "gmm" | "percentile" | "exemplar"
    thresholds: list[float]  # [unusable_upper, recoverable_upper]
    confidence: float  # GVF score (0-1)
    dip_test_p: float  # Hartigan p-value
    gmm_means: list[float] = []
    explanation: str
    strictness: float


class ImageVerdictRecord(BaseModel):
    image_id: str
    image_path: str
    verdict: str  # "usable" | "recoverable" | "unusable" | "error"
    scores: dict[str, float]
    failed_dimensions: list[str]
    explanation: str


class RecalibrationResult(BaseModel):
    dimension_calibrations: dict[str, DimensionCalibration]
    image_verdicts: dict[str, ImageVerdictRecord]
    method_summary: str
    overall_confidence: float

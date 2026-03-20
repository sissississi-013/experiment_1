import math
from pydantic import BaseModel


class ToolCalibration(BaseModel):
    tool_name: str
    raw_good_scores: list[float]
    raw_bad_scores: list[float]
    platt_a: float = 0.0
    platt_b: float = 0.0
    calibrated_threshold: float = 0.5
    separability: float = 0.0

    def apply_platt(self, raw_score: float) -> float:
        return 1.0 / (1.0 + math.exp(self.platt_a * raw_score + self.platt_b))


class EmbeddingRecord(BaseModel):
    image_path: str
    embedding: list[float]
    label: str


class ThresholdExplanation(BaseModel):
    dimension: str
    threshold: float
    explanation: str


class CalibrationResult(BaseModel):
    tool_calibrations: dict[str, ToolCalibration]
    exemplar_embeddings: list[EmbeddingRecord]
    threshold_report: list[ThresholdExplanation]

from abc import ABC, abstractmethod
from typing import Any
from PIL import Image
from validation_pipeline.schemas.execution import ToolResult
from validation_pipeline.schemas.calibration import ToolCalibration


class BaseTool(ABC):
    name: str
    task_type: str
    tier: int
    input_type: str = "image"
    output_type: str = "float"
    source: str = "local"
    catalog_id: str | None = None

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def execute(self, image: Image.Image, **kwargs) -> Any:
        pass

    @abstractmethod
    def normalize(self, raw_output: Any, calibration: ToolCalibration | None = None) -> ToolResult:
        pass

import cv2
import numpy as np
from typing import Any
from PIL import Image
from validation_pipeline.tools.base import BaseTool
from validation_pipeline.schemas.execution import ToolResult
from validation_pipeline.schemas.calibration import ToolCalibration


class LaplacianBlurTool(BaseTool):
    name = "laplacian_blur"
    task_type = "image_quality"
    tier = 1
    output_type = "float"
    source = "local"

    def __init__(self, config: dict = {}):
        super().__init__(config)

    def execute(self, image: Image.Image, **kwargs) -> float:
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def normalize(self, raw_output: Any, calibration: ToolCalibration | None = None) -> ToolResult:
        if calibration:
            score = calibration.apply_platt(raw_output)
        else:
            score = min(raw_output / 1000.0, 1.0)
        return ToolResult(
            tool_name=self.name,
            dimension="blur",
            score=score,
            passed=True,
            threshold=0.0,
            raw_output=raw_output,
            calibration_method="platt" if calibration else "default",
        )


class HistogramExposureTool(BaseTool):
    name = "histogram_exposure"
    task_type = "image_quality"
    tier = 1
    output_type = "float"
    source = "local"

    def __init__(self, config: dict = {}):
        super().__init__(config)

    def execute(self, image: Image.Image, **kwargs) -> float:
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return float(np.mean(gray)) / 255.0

    def normalize(self, raw_output: Any, calibration: ToolCalibration | None = None) -> ToolResult:
        if calibration:
            score = calibration.apply_platt(raw_output)
        else:
            score = raw_output
        return ToolResult(
            tool_name=self.name,
            dimension="exposure",
            score=score,
            passed=True,
            threshold=0.0,
            raw_output=raw_output,
            calibration_method="platt" if calibration else "default",
        )


class PixelStatsTool(BaseTool):
    name = "pixel_stats"
    task_type = "image_quality"
    tier = 1
    output_type = "float"
    source = "local"

    def __init__(self, config: dict = {}):
        super().__init__(config)

    def execute(self, image: Image.Image, **kwargs) -> float:
        arr = np.array(image).astype(np.float32)
        return float(np.std(arr))

    def normalize(self, raw_output: Any, calibration: ToolCalibration | None = None) -> ToolResult:
        score = min(raw_output / 80.0, 1.0)
        return ToolResult(
            tool_name=self.name,
            dimension="information_content",
            score=score,
            passed=True,
            threshold=0.0,
            raw_output=raw_output,
            calibration_method="default",
        )

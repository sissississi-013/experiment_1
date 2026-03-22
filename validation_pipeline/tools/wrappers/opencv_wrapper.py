import cv2
import numpy as np
from typing import Any
from PIL import Image
from validation_pipeline.tools.base import BaseTool
from validation_pipeline.schemas.execution import ToolResult
from validation_pipeline.schemas.calibration import ToolCalibration


class LaplacianBlurTool(BaseTool):
    """Sharpness detection using Laplacian variance.

    Higher Laplacian variance = sharper image. Typical ranges:
    - <100: very blurry
    - 100-500: somewhat soft
    - 500-2000: reasonably sharp
    - >2000: very sharp

    Normalization uses a sigmoid to give meaningful spread across the range,
    not just min(raw/1000, 1.0) which saturates everything.
    """
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
            # Sigmoid normalization centered at 500 (transition from soft to sharp)
            # Gives 0.27 at raw=100, 0.50 at raw=500, 0.73 at raw=1000, 0.95 at raw=3000
            score = 1.0 / (1.0 + np.exp(-(raw_output - 500) / 400))
            score = float(score)
        return ToolResult(
            tool_name=self.name,
            dimension="blur",
            score=score,
            passed=True,
            threshold=0.0,
            raw_output=raw_output,
            explanation=f"Laplacian variance: {raw_output:.0f} ({'very sharp' if raw_output > 2000 else 'sharp' if raw_output > 500 else 'soft' if raw_output > 100 else 'blurry'})",
            calibration_method="platt" if calibration else "sigmoid",
        )


class HistogramExposureTool(BaseTool):
    """Exposure analysis using histogram statistics.

    Measures both the mean brightness AND the spread. Good exposure means
    the histogram uses a wide range (not clipped to dark or bright).

    Score combines:
    - Brightness: how close mean is to 0.5 (middle gray)
    - Contrast: standard deviation of pixel values (higher = more dynamic range)
    """
    name = "histogram_exposure"
    task_type = "image_quality"
    tier = 1
    output_type = "float"
    source = "local"

    def __init__(self, config: dict = {}):
        super().__init__(config)

    def execute(self, image: Image.Image, **kwargs) -> dict:
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        # Check for clipping (too many pure black or pure white pixels)
        clip_dark = float(np.mean(gray < 0.02))  # % of near-black pixels
        clip_bright = float(np.mean(gray > 0.98))  # % of near-white pixels

        return {
            "mean": mean_brightness,
            "std": std_brightness,
            "clip_dark": clip_dark,
            "clip_bright": clip_bright,
        }

    def normalize(self, raw_output: Any, calibration: ToolCalibration | None = None) -> ToolResult:
        if isinstance(raw_output, dict):
            mean_b = raw_output["mean"]
            std_b = raw_output["std"]
            clip_d = raw_output.get("clip_dark", 0)
            clip_b = raw_output.get("clip_bright", 0)
        else:
            # Backward compatibility with old float format
            mean_b = float(raw_output)
            std_b = 0.2
            clip_d = 0
            clip_b = 0

        if calibration:
            score = calibration.apply_platt(mean_b)
        else:
            # Score = brightness quality (how close to 0.5) * contrast quality
            brightness_score = 1.0 - abs(mean_b - 0.5) * 2  # 1.0 at 0.5, 0.0 at 0/1
            contrast_score = min(std_b / 0.25, 1.0)  # Good contrast > 0.2 std
            clip_penalty = max(0, 1.0 - (clip_d + clip_b) * 5)  # Penalize clipping
            score = float(brightness_score * 0.5 + contrast_score * 0.3 + clip_penalty * 0.2)
            score = max(0.0, min(1.0, score))

        # Build explanation
        if mean_b < 0.2:
            exp_desc = "very dark/underexposed"
        elif mean_b < 0.35:
            exp_desc = "slightly dark"
        elif mean_b > 0.8:
            exp_desc = "very bright/overexposed"
        elif mean_b > 0.65:
            exp_desc = "slightly bright"
        else:
            exp_desc = "well-exposed"

        return ToolResult(
            tool_name=self.name,
            dimension="exposure",
            score=score,
            passed=True,
            threshold=0.0,
            raw_output=raw_output,
            explanation=f"Mean brightness: {mean_b:.2f}, contrast: {std_b:.2f} ({exp_desc})",
            calibration_method="platt" if calibration else "composite",
        )


class PixelStatsTool(BaseTool):
    """Information content detection using pixel statistics.

    Detects garbage frames: solid colors, blank images, near-uniform images.
    Higher std = more visual information.
    """
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
            explanation=f"Pixel std: {raw_output:.1f} ({'rich content' if raw_output > 60 else 'moderate' if raw_output > 30 else 'low information'})",
            calibration_method="default",
        )

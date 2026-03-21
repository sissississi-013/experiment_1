import io
import os
import time
import requests
from typing import Any
from PIL import Image
from validation_pipeline.tools.base import BaseTool
from validation_pipeline.schemas.execution import ToolResult
from validation_pipeline.schemas.calibration import ToolCalibration


class RoboflowObjectDetectionTool(BaseTool):
    name = "roboflow_object_detection"
    task_type = "content_detection"
    tier = 2
    output_type = "dict"
    source = "api"

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.api_key_env = config.get("api_key_env", "ROBOFLOW_API_KEY")
        self.model = config.get("model", "coco/1")
        self.max_retries = 3
        self.timeout = 10

    def execute(self, image: Image.Image, **kwargs) -> dict:
        target_label = kwargs.get("target_label", "")
        api_key = os.environ.get(self.api_key_env, "")

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        url = f"https://detect.roboflow.com/{self.model}"
        params = {"api_key": api_key}

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    url, params=params, files={"file": ("image.jpg", img_bytes)},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                if attempt == self.max_retries - 1:
                    return {"best_confidence": 0.0, "detections": [], "target_label": target_label}
                time.sleep(2 ** attempt)

        detections = data.get("predictions", [])
        matching = [d for d in detections if d.get("class", "").lower() == target_label.lower()]
        best_confidence = max((d["confidence"] for d in matching), default=0.0)

        return {
            "best_confidence": best_confidence,
            "detections": detections,
            "target_label": target_label,
        }

    def normalize(self, raw_output: Any, calibration: ToolCalibration | None = None) -> ToolResult:
        score = raw_output["best_confidence"]
        target = raw_output.get("target_label", "object")
        if calibration:
            score = calibration.apply_platt(score)
        return ToolResult(
            tool_name=self.name,
            dimension="content",
            score=score,
            passed=True,
            threshold=0.0,
            raw_output=raw_output,
            explanation=f"{target} detected with {score:.2f} confidence" if score > 0 else f"{target} not detected",
            calibration_method="platt" if calibration else "default",
        )

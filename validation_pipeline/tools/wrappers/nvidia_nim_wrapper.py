import io
import os
import time
import base64
import requests
from typing import Any
from PIL import Image
from validation_pipeline.tools.base import BaseTool
from validation_pipeline.schemas.execution import ToolResult
from validation_pipeline.schemas.calibration import ToolCalibration


INFERENCE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino"


class NVIDIAGroundingDINOTool(BaseTool):
    name = "nvidia_grounding_dino"
    task_type = "content_detection"
    tier = 2
    output_type = "dict"
    source = "api"

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.api_key_env = config.get("api_key_env", "NVIDIA_NIM_API_KEY")
        self.max_retries = 3
        self.timeout = 60
        self.detection_threshold = config.get("detection_threshold", 0.3)

    def _encode_image(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _run_inference(self, b64_image: str, target_label: str) -> dict:
        """Send inference request with inline base64 image."""
        api_key = os.environ.get(self.api_key_env, "")

        # Singularize simple plurals for better detection
        label = target_label.rstrip("s") if target_label.endswith("s") and not target_label.endswith("ss") else target_label
        prompt = f"a {label} ."

        resp = requests.post(
            INFERENCE_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": "Grounding-Dino",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "media_url", "media_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }},
                    ],
                }],
                "threshold": self.detection_threshold,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def _parse_response(self, data: dict) -> dict:
        """Parse NIM GroundingDINO JSON response."""
        detections = []

        for choice in data.get("choices", []):
            content = choice.get("message", {}).get("content", {})
            if isinstance(content, str):
                continue
            for bb in content.get("boundingBoxes", []):
                phrase = bb.get("phrase", "")
                bboxes = bb.get("bboxes", [])
                confidences = bb.get("confidence", [])
                for i, bbox in enumerate(bboxes):
                    conf = confidences[i] if i < len(confidences) else 0.0
                    detections.append({
                        "confidence": float(conf),
                        "label": phrase,
                        "bbox": bbox,
                    })

        best = max((d["confidence"] for d in detections), default=0.0)
        return {"best_confidence": best, "detections": detections}

    def execute(self, image: Image.Image, **kwargs) -> dict:
        target_label = kwargs.get("target_label", "")
        b64_image = self._encode_image(image)

        for attempt in range(self.max_retries):
            try:
                result = self._run_inference(b64_image, target_label)
                result["target_label"] = target_label
                return result
            except Exception:
                if attempt == self.max_retries - 1:
                    return {"best_confidence": 0.0, "detections": [], "target_label": target_label}
                time.sleep(2 ** attempt)

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

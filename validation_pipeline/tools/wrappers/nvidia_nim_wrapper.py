import io
import os
import time
import zipfile
import requests
import numpy as np
from typing import Any
from PIL import Image
from validation_pipeline.tools.base import BaseTool
from validation_pipeline.schemas.execution import ToolResult
from validation_pipeline.schemas.calibration import ToolCalibration


ASSET_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
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
        self.timeout = 30
        self.detection_threshold = config.get("detection_threshold", 0.3)

    def _get_api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    def _upload_asset(self, image: Image.Image) -> str:
        """Upload image to NVIDIA asset store, return asset_id."""
        api_key = self._get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        # Step 1: Create asset upload URL
        resp = requests.post(
            ASSET_URL,
            headers=headers,
            json={"contentType": "image/jpeg", "description": "input image"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        asset_data = resp.json()
        upload_url = asset_data["uploadUrl"]
        asset_id = asset_data["assetId"]

        # Step 2: Upload image bytes to presigned URL
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        requests.put(
            upload_url,
            data=img_bytes,
            headers={
                "Content-Type": "image/jpeg",
                "x-amz-meta-nvcf-asset-description": "input image",
            },
            timeout=self.timeout,
        )

        return asset_id

    def _run_inference(self, asset_id: str, target_label: str) -> dict:
        """Send inference request and parse response."""
        api_key = self._get_api_key()

        # Format prompt: GroundingDINO expects "a horse ." format
        prompt = f"a {target_label} ."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "NVCF-INPUT-ASSET-REFERENCES": asset_id,
            "NVCF-FUNCTION-ASSET-IDS": asset_id,
        }

        body = {
            "model": "Grounding-Dino",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "media_url", "media_url": {
                            "url": f"data:image/jpeg;asset_id,{asset_id}"
                        }},
                    ],
                }
            ],
            "threshold": self.detection_threshold,
        }

        resp = requests.post(
            INFERENCE_URL,
            headers=headers,
            json=body,
            timeout=self.timeout,
        )

        # Handle async (202) — poll for result
        if resp.status_code == 202:
            req_id = resp.headers.get("NVCF-REQID", "")
            return self._poll_result(req_id)

        resp.raise_for_status()
        return self._parse_response(resp.content, prompt)

    def _poll_result(self, req_id: str) -> dict:
        """Poll for async result."""
        api_key = self._get_api_key()
        poll_url = f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{req_id}"
        headers = {"Authorization": f"Bearer {api_key}", "accept": "application/json"}

        for _ in range(10):
            time.sleep(1)
            resp = requests.get(poll_url, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                return self._parse_response(resp.content, "")
        return {"best_confidence": 0.0, "detections": []}

    def _parse_response(self, content: bytes, prompt: str) -> dict:
        """Parse ZIP response with pred_logits and pred_boxes tensors."""
        try:
            buf = io.BytesIO(content)
            if not zipfile.is_zipfile(buf):
                # Try parsing as JSON (some endpoints return JSON directly)
                import json
                try:
                    data = json.loads(content)
                    return self._parse_json_response(data)
                except (json.JSONDecodeError, ValueError):
                    return {"best_confidence": 0.0, "detections": []}

            buf.seek(0)
            detections = []
            with zipfile.ZipFile(buf, "r") as zf:
                names = zf.namelist()

                logits = None
                boxes = None
                for name in names:
                    raw = zf.read(name)
                    if "logit" in name.lower():
                        logits = np.frombuffer(raw, dtype=np.float32)
                    elif "box" in name.lower():
                        boxes = np.frombuffer(raw, dtype=np.float32)

                if logits is not None and boxes is not None:
                    # pred_logits: (B, 900), pred_boxes: (B, 900, 4)
                    num_queries = 900
                    if len(logits) >= num_queries:
                        logits = logits[:num_queries]
                    if len(boxes) >= num_queries * 4:
                        boxes = boxes[:num_queries * 4].reshape(-1, 4)

                    # Filter by threshold
                    mask = logits > self.detection_threshold
                    for i in np.where(mask)[0]:
                        cx, cy, w, h = boxes[i]
                        detections.append({
                            "confidence": float(logits[i]),
                            "bbox_cxcywh": [float(cx), float(cy), float(w), float(h)],
                        })

            best = max((d["confidence"] for d in detections), default=0.0)
            return {"best_confidence": best, "detections": detections}

        except Exception:
            return {"best_confidence": 0.0, "detections": []}

    def _parse_json_response(self, data: dict) -> dict:
        """Parse JSON response format (some NIM versions return this)."""
        detections = []
        if "predictions" in data:
            for pred in data["predictions"]:
                detections.append({
                    "confidence": pred.get("confidence", 0.0),
                    "label": pred.get("label", ""),
                    "bbox": pred.get("bbox", []),
                })
        elif "choices" in data:
            # OpenAI-compatible format
            for choice in data["choices"]:
                content = choice.get("message", {}).get("content", "")
                if content:
                    detections.append({"confidence": 1.0, "label": content})

        best = max((d["confidence"] for d in detections), default=0.0)
        return {"best_confidence": best, "detections": detections}

    def execute(self, image: Image.Image, **kwargs) -> dict:
        target_label = kwargs.get("target_label", "")

        for attempt in range(self.max_retries):
            try:
                asset_id = self._upload_asset(image)
                result = self._run_inference(asset_id, target_label)
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

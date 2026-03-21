import os
import io
import base64
import instructor
from typing import Any
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel
from validation_pipeline.tools.base import BaseTool
from validation_pipeline.schemas.execution import ToolResult
from validation_pipeline.schemas.calibration import ToolCalibration
from validation_pipeline.retry import retry_with_policy
from validation_pipeline.config import RetryPolicy
from validation_pipeline.errors import ToolError


class VLMResult(BaseModel):
    score: float
    justification: str


class GPT4VisionTool(BaseTool):
    name = "gpt4o_vision_semantic"
    task_type = "semantic_quality"
    tier = 3
    output_type = "dict"
    source = "api"

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.model = config.get("model", "gpt-4o")
        self.max_tokens = config.get("max_tokens", 100)
        self.timeout = 30

    def _encode_image(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def execute(self, image: Image.Image, **kwargs) -> dict:
        semantic_question = kwargs.get("semantic_question", "overall quality")
        api_key = os.environ.get(self.api_key_env, "")
        b64_image = self._encode_image(image)
        client = instructor.from_openai(OpenAI(api_key=api_key))

        def _call():
            result = client.chat.completions.create(
                model=self.model,
                response_model=VLMResult,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            f"Rate this image on: \"{semantic_question}\". "
                            "Return a score from 0.0 (worst) to 1.0 (best) "
                            "and a one-sentence justification."
                        )},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }},
                    ]},
                ],
            )
            return {
                "score": result.score,
                "justification": result.justification,
                "semantic_question": semantic_question,
            }

        return retry_with_policy(
            fn=_call,
            policy=RetryPolicy(),
            error_cls=ToolError,
            module=self.name,
            context={"semantic_question": semantic_question},
        )

    def normalize(self, raw_output: Any, calibration: ToolCalibration | None = None) -> ToolResult:
        score = raw_output["score"]
        if calibration:
            score = calibration.apply_platt(score)
        return ToolResult(
            tool_name=self.name,
            dimension="semantic",
            score=score,
            passed=True,
            threshold=0.0,
            raw_output=raw_output,
            explanation=raw_output.get("justification", ""),
            calibration_method="platt" if calibration else "default",
        )

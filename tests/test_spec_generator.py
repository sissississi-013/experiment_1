import pytest
from unittest.mock import patch
from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.schemas.spec import FormalSpec
from validation_pipeline.modules.spec_generator import generate_spec
from validation_pipeline.errors import LLMError


def test_generate_spec_returns_formal_spec():
    user_input = UserInput(
        dataset_path="/data/coco",
        intent="I want high-quality images of horses, no blur, good lighting",
    )
    with patch("validation_pipeline.modules.spec_generator._call_llm") as mock_llm:
        mock_llm.return_value = FormalSpec(
            restated_request="Find horse images that are sharp and well-lit",
            assumptions=["COCO dataset", "RGB images"],
            content_criteria=[{"object_or_scene": "horse", "must_contain": True, "exemplar_based": False}],
            quality_criteria=[
                {"dimension": "blur", "description": "images must not be blurry"},
                {"dimension": "exposure", "description": "images must have proper lighting"},
            ],
            quantity_targets={"min_images": None, "per_class": False},
            output_format={"format": "json", "include_rejected": True, "include_recoverable": True},
            success_criteria="Images contain horses, are not blurry, and are well-lit",
        )
        spec = generate_spec(user_input)
        assert isinstance(spec, FormalSpec)
        assert len(spec.content_criteria) >= 1
        assert len(spec.quality_criteria) >= 1
        assert spec.user_confirmed is False


def test_spec_generator_raises_llm_error():
    ui = UserInput(dataset_path="/data", intent="test")
    with patch("validation_pipeline.modules.spec_generator._call_llm", side_effect=Exception("LLM failed")):
        with pytest.raises(LLMError) as exc_info:
            generate_spec(ui)
    assert exc_info.value.module == "spec_generator"

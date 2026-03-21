import pytest
from unittest.mock import patch
from validation_pipeline.errors import LLMError
from validation_pipeline.modules.planner import generate_plan, SYSTEM_PROMPT
from validation_pipeline.schemas.spec import FormalSpec, QualityCriterion, QuantityTarget, OutputFormat
from validation_pipeline.schemas.calibration import CalibrationResult


def test_planner_prompt_includes_tool_params_rules():
    assert "tool_params" in SYSTEM_PROMPT
    assert "target_label" in SYSTEM_PROMPT
    assert "semantic_question" in SYSTEM_PROMPT


def test_planner_raises_llm_error():
    spec = FormalSpec(
        restated_request="test", assumptions=[], content_criteria=[],
        quality_criteria=[], quantity_targets=QuantityTarget(),
        output_format=OutputFormat(), success_criteria="test",
    )
    cal = CalibrationResult(tool_calibrations={}, exemplar_embeddings=[], threshold_report=[])
    with patch("validation_pipeline.modules.planner._call_llm", side_effect=Exception("LLM failed")):
        with pytest.raises(LLMError) as exc_info:
            generate_plan(spec, cal, [])
    assert exc_info.value.module == "planner"

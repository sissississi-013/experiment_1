from validation_pipeline.modules.planner import SYSTEM_PROMPT


def test_planner_prompt_includes_tool_params_rules():
    assert "tool_params" in SYSTEM_PROMPT
    assert "target_label" in SYSTEM_PROMPT
    assert "semantic_question" in SYSTEM_PROMPT

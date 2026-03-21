import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from validation_pipeline.tools.wrappers.openai_vision_wrapper import GPT4VisionTool


def _make_mock_client(score=0.85, justification="Good lighting"):
    mock_result = MagicMock()
    mock_result.score = score
    mock_result.justification = justification
    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_result
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat
    return mock_client


def test_gpt4v_execute():
    tool = GPT4VisionTool({"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o", "max_tokens": 100})
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    mock_client = _make_mock_client(0.85, "Natural daylight, good exposure")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
         patch("instructor.from_openai", return_value=mock_client):
        result = tool.execute(img, semantic_question="natural lighting")
    assert result["score"] == 0.85
    assert result["justification"] == "Natural daylight, good exposure"


def test_gpt4v_execute_low_score():
    tool = GPT4VisionTool({"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o", "max_tokens": 100})
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    mock_client = _make_mock_client(0.2, "Very dark, underexposed")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
         patch("instructor.from_openai", return_value=mock_client):
        result = tool.execute(img, semantic_question="natural lighting")
    assert result["score"] == 0.2


def test_gpt4v_normalize():
    tool = GPT4VisionTool({"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o", "max_tokens": 100})
    raw = {"score": 0.85, "justification": "Natural daylight", "semantic_question": "natural lighting"}
    tr = tool.normalize(raw)
    assert tr.score == 0.85
    assert tr.dimension == "semantic"
    assert "Natural daylight" in tr.explanation


def test_gpt4v_retry_on_failure():
    tool = GPT4VisionTool({"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o", "max_tokens": 100})
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    mock_client = MagicMock()
    call_count = {"n": 0}
    def side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("API timeout")
        result = MagicMock()
        result.score = 0.7
        result.justification = "Recovered"
        return result
    mock_client.chat.completions.create.side_effect = side_effect
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
         patch("instructor.from_openai", return_value=mock_client), \
         patch("time.sleep"):
        result = tool.execute(img, semantic_question="test")
    assert result["score"] == 0.7


import pytest
from validation_pipeline.errors import ToolError

def test_gpt4v_raises_tool_error_on_exhaustion():
    from validation_pipeline.tools.wrappers.openai_vision_wrapper import GPT4VisionTool
    tool = GPT4VisionTool({"api_key_env": "OPENAI_API_KEY", "model": "gpt-4o", "max_tokens": 100})
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("always fails")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
         patch("instructor.from_openai", return_value=mock_client), \
         patch("time.sleep"):
        with pytest.raises(ToolError) as exc_info:
            tool.execute(img, semantic_question="test")
    assert exc_info.value.module == "gpt4o_vision_semantic"

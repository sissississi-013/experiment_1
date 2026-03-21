import json
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from validation_pipeline.tools.wrappers.roboflow_wrapper import RoboflowObjectDetectionTool


def test_roboflow_execute_finds_target():
    tool = RoboflowObjectDetectionTool({"api_key_env": "ROBOFLOW_API_KEY", "model": "coco/1"})
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "predictions": [
            {"class": "horse", "confidence": 0.92, "x": 50, "y": 50, "width": 80, "height": 60},
            {"class": "person", "confidence": 0.75, "x": 20, "y": 20, "width": 30, "height": 50},
        ]
    }

    with patch("requests.post", return_value=mock_response), \
         patch.dict("os.environ", {"ROBOFLOW_API_KEY": "test-key"}):
        result = tool.execute(img, target_label="horse")

    assert result["best_confidence"] == 0.92
    assert len(result["detections"]) == 2


def test_roboflow_execute_target_not_found():
    tool = RoboflowObjectDetectionTool({"api_key_env": "ROBOFLOW_API_KEY", "model": "coco/1"})
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "predictions": [
            {"class": "person", "confidence": 0.85, "x": 50, "y": 50, "width": 30, "height": 50},
        ]
    }

    with patch("requests.post", return_value=mock_response), \
         patch.dict("os.environ", {"ROBOFLOW_API_KEY": "test-key"}):
        result = tool.execute(img, target_label="horse")

    assert result["best_confidence"] == 0.0


def test_roboflow_normalize():
    tool = RoboflowObjectDetectionTool({"api_key_env": "ROBOFLOW_API_KEY", "model": "coco/1"})
    raw = {"best_confidence": 0.87, "detections": [{"class": "horse", "confidence": 0.87}], "target_label": "horse"}
    tr = tool.normalize(raw)
    assert tr.score == 0.87
    assert tr.dimension == "content"
    assert "horse" in tr.explanation


def test_roboflow_retry_on_failure():
    tool = RoboflowObjectDetectionTool({"api_key_env": "ROBOFLOW_API_KEY", "model": "coco/1"})
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    fail_resp = MagicMock()
    fail_resp.status_code = 500
    fail_resp.raise_for_status.side_effect = Exception("Server error")

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = {"predictions": [{"class": "horse", "confidence": 0.9, "x": 0, "y": 0, "width": 10, "height": 10}]}

    with patch("requests.post", side_effect=[fail_resp, ok_resp]), \
         patch.dict("os.environ", {"ROBOFLOW_API_KEY": "test-key"}), \
         patch("time.sleep"):
        result = tool.execute(img, target_label="horse")

    assert result["best_confidence"] == 0.9

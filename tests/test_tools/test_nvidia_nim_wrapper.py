import json
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from validation_pipeline.tools.wrappers.nvidia_nim_wrapper import NVIDIAGroundingDINOTool


def _make_tool():
    return NVIDIAGroundingDINOTool({"api_key_env": "NVIDIA_NIM_API_KEY", "detection_threshold": 0.3})


def _mock_response(detections):
    """Create a mock NIM GroundingDINO response."""
    bounding_boxes = []
    for d in detections:
        # Group by phrase
        found = False
        for bb in bounding_boxes:
            if bb["phrase"] == d.get("label", ""):
                bb["bboxes"].append(d.get("bbox", [0, 0, 10, 10]))
                bb["confidence"].append(d["confidence"])
                found = True
                break
        if not found:
            bounding_boxes.append({
                "phrase": d.get("label", ""),
                "bboxes": [d.get("bbox", [0, 0, 10, 10])],
                "confidence": [d["confidence"]],
            })

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": {
                    "frameNo": 0,
                    "frameWidth": 640,
                    "frameHeight": 480,
                    "message": "a horse . .",
                    "boundingBoxes": bounding_boxes,
                },
            },
        }],
    }
    resp.raise_for_status = MagicMock()
    return resp


def test_nvidia_dino_execute_detects_target():
    tool = _make_tool()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    mock_resp = _mock_response([
        {"confidence": 0.88, "label": "a horse", "bbox": [337, 153, 202, 200]},
    ])

    with patch("requests.post", return_value=mock_resp), \
         patch.dict("os.environ", {"NVIDIA_NIM_API_KEY": "nvapi-test"}):
        result = tool.execute(img, target_label="horse")

    assert result["best_confidence"] == 0.88
    assert result["target_label"] == "horse"
    assert len(result["detections"]) == 1


def test_nvidia_dino_execute_no_detection():
    tool = _make_tool()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    mock_resp = _mock_response([])

    with patch("requests.post", return_value=mock_resp), \
         patch.dict("os.environ", {"NVIDIA_NIM_API_KEY": "nvapi-test"}):
        result = tool.execute(img, target_label="horse")

    assert result["best_confidence"] == 0.0


def test_nvidia_dino_normalize():
    tool = _make_tool()
    raw = {"best_confidence": 0.88, "detections": [{"confidence": 0.88}], "target_label": "horse"}
    tr = tool.normalize(raw)
    assert tr.score == 0.88
    assert tr.dimension == "content"
    assert "horse" in tr.explanation


def test_nvidia_dino_singularizes_plural_label():
    tool = _make_tool()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    captured_body = {}

    def mock_post(url, **kwargs):
        captured_body.update(kwargs.get("json", {}))
        return _mock_response([{"confidence": 0.7, "label": "a horse", "bbox": [0, 0, 10, 10]}])

    with patch("requests.post", side_effect=mock_post), \
         patch.dict("os.environ", {"NVIDIA_NIM_API_KEY": "nvapi-test"}):
        tool.execute(img, target_label="horses")

    # Should have singularized "horses" to "horse" in the prompt
    prompt = captured_body["messages"][0]["content"][0]["text"]
    assert prompt == "a horse ."


def test_nvidia_dino_retry_on_failure():
    tool = _make_tool()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    call_count = {"n": 0}

    def mock_post(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("Network error")
        return _mock_response([{"confidence": 0.75, "label": "a horse", "bbox": [0, 0, 10, 10]}])

    with patch("requests.post", side_effect=mock_post), \
         patch.dict("os.environ", {"NVIDIA_NIM_API_KEY": "nvapi-test"}), \
         patch("time.sleep"):
        result = tool.execute(img, target_label="horse")

    assert result["best_confidence"] == 0.75

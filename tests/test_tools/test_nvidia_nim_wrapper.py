import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from validation_pipeline.tools.wrappers.nvidia_nim_wrapper import NVIDIAGroundingDINOTool


def _make_tool():
    return NVIDIAGroundingDINOTool({"api_key_env": "NVIDIA_NIM_API_KEY", "detection_threshold": 0.3})


def test_nvidia_dino_execute_detects_target():
    tool = _make_tool()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    # Mock asset upload
    mock_asset_resp = MagicMock()
    mock_asset_resp.status_code = 200
    mock_asset_resp.json.return_value = {
        "uploadUrl": "https://fake-upload-url.com",
        "assetId": "test-asset-123",
    }
    mock_asset_resp.raise_for_status = MagicMock()

    mock_upload_resp = MagicMock()
    mock_upload_resp.status_code = 200

    # Mock inference — return JSON with detections
    import json
    mock_infer_resp = MagicMock()
    mock_infer_resp.status_code = 200
    mock_infer_resp.content = json.dumps({
        "predictions": [
            {"confidence": 0.88, "label": "horse", "bbox": [10, 20, 80, 60]},
        ]
    }).encode()
    mock_infer_resp.raise_for_status = MagicMock()
    mock_infer_resp.headers = {}

    with patch("requests.post", side_effect=[mock_asset_resp, mock_infer_resp]), \
         patch("requests.put", return_value=mock_upload_resp), \
         patch.dict("os.environ", {"NVIDIA_NIM_API_KEY": "nvapi-test"}):
        result = tool.execute(img, target_label="horse")

    assert result["best_confidence"] == 0.88
    assert result["target_label"] == "horse"


def test_nvidia_dino_execute_no_detection():
    tool = _make_tool()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    mock_asset_resp = MagicMock()
    mock_asset_resp.status_code = 200
    mock_asset_resp.json.return_value = {
        "uploadUrl": "https://fake-upload-url.com",
        "assetId": "test-asset-456",
    }
    mock_asset_resp.raise_for_status = MagicMock()

    mock_upload_resp = MagicMock()

    import json
    mock_infer_resp = MagicMock()
    mock_infer_resp.status_code = 200
    mock_infer_resp.content = json.dumps({"predictions": []}).encode()
    mock_infer_resp.raise_for_status = MagicMock()
    mock_infer_resp.headers = {}

    with patch("requests.post", side_effect=[mock_asset_resp, mock_infer_resp]), \
         patch("requests.put", return_value=mock_upload_resp), \
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


def test_nvidia_dino_retry_on_failure():
    tool = _make_tool()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

    call_count = {"n": 0}

    def mock_post(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:  # First two calls (asset upload + inference) fail
            raise Exception("Network error")
        # Second attempt succeeds
        if call_count["n"] == 3:  # Asset upload
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"uploadUrl": "https://fake.com", "assetId": "retry-asset"}
            resp.raise_for_status = MagicMock()
            return resp
        else:  # Inference
            import json
            resp = MagicMock()
            resp.status_code = 200
            resp.content = json.dumps({"predictions": [{"confidence": 0.75, "label": "horse"}]}).encode()
            resp.raise_for_status = MagicMock()
            resp.headers = {}
            return resp

    with patch("requests.post", side_effect=mock_post), \
         patch("requests.put", return_value=MagicMock()), \
         patch.dict("os.environ", {"NVIDIA_NIM_API_KEY": "nvapi-test"}), \
         patch("time.sleep"):
        result = tool.execute(img, target_label="horse")

    assert result["best_confidence"] == 0.75

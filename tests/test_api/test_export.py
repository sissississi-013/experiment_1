import json, zipfile, io, pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from api.server import app

@pytest.fixture
def client():
    app.state.config = MagicMock()
    app.state.store = MagicMock()
    return TestClient(app)

def test_export_returns_zip(client, tmp_path):
    img_path = tmp_path / "img1.jpg"
    img_path.write_bytes(b"\xff\xd8fake")
    app.state.store.get_run.return_value = {"id": "abc", "status": "completed", "intent": "test"}
    app.state.store.get_run_images.return_value = [{"image_id": "img1", "image_path": str(img_path), "verdict": "usable", "scores": {"blur": 0.9}}]
    resp = client.get("/api/runs/abc/export?filter=usable")
    assert resp.status_code == 200
    assert "application/zip" in resp.headers["content-type"]
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    assert "manifest.json" in z.namelist()
    assert any(n.endswith(".jpg") for n in z.namelist())

def test_export_not_found(client):
    app.state.store.get_run.return_value = None
    resp = client.get("/api/runs/x/export")
    assert resp.status_code == 404

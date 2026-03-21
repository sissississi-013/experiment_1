import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from api.server import app

@pytest.fixture
def client():
    app.state.config = MagicMock()
    app.state.store = MagicMock()
    return TestClient(app)

def test_get_run_images(client):
    app.state.store.get_run_images.return_value = [{"image_id": "img1", "verdict": "usable"}]
    resp = client.get("/api/runs/abc/images")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

def test_get_run_images_filtered(client):
    app.state.store.get_run_images.return_value = []
    resp = client.get("/api/runs/abc/images?verdict=usable")
    assert resp.status_code == 200
    app.state.store.get_run_images.assert_called_with("abc", verdict="usable")

def test_query_images(client):
    app.state.store.query_images.return_value = [{"image_id": "img1"}]
    resp = client.get("/api/images?verdict=usable&limit=10")
    assert resp.status_code == 200

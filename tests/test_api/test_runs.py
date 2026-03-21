import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.server import app

@pytest.fixture
def client():
    app.state.config = MagicMock()
    app.state.store = MagicMock()
    return TestClient(app)

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200

def test_list_runs(client):
    app.state.store.list_runs.return_value = [
        {"id": "abc", "intent": "test", "status": "completed"},
    ]
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

def test_get_run(client):
    app.state.store.get_run.return_value = {"id": "abc", "intent": "test"}
    resp = client.get("/api/runs/abc")
    assert resp.status_code == 200
    assert resp.json()["id"] == "abc"

def test_get_run_not_found(client):
    app.state.store.get_run.return_value = None
    resp = client.get("/api/runs/nonexistent")
    assert resp.status_code == 404

def test_create_run(client):
    app.state.store.create_run.return_value = "new-id"
    with patch("api.routes.runs.run_pipeline_background"):
        resp = client.post("/api/runs", json={"intent": "find horses", "dataset_description": "10 horses from COCO"})
    assert resp.status_code == 201
    assert "run_id" in resp.json()

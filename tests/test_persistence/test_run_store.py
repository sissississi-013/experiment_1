"""Tests for RunStore using mocked HTTP requests."""
import json
from unittest.mock import patch, MagicMock
from validation_pipeline.persistence.run_store import RunStore


def _mock_store():
    """Create a RunStore with mocked HTTP calls."""
    store = RunStore("postgresql://user:pass@fake-host.neon.tech/neondb?sslmode=require")
    return store


def _mock_response(rows=None, fields=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "fields": fields or [],
        "rows": rows or [],
        "rowCount": len(rows) if rows else 0,
    }
    resp.raise_for_status = MagicMock()
    return resp


def test_run_store_parses_connection_string():
    store = RunStore("postgresql://user:pass@my-host.neon.tech/neondb?sslmode=require")
    assert store.sql_url == "https://my-host.neon.tech/sql"


def test_create_run():
    store = _mock_store()
    with patch("requests.post", return_value=_mock_response()) as mock_post:
        store.create_run(run_id="abc123", intent="find horses", config_json={"model": "gpt-4o"}, dataset_path="/data")
    call_body = mock_post.call_args[1]["json"]
    assert "INSERT INTO runs" in call_body["query"]
    assert "abc123" in call_body["params"]


def test_complete_run():
    store = _mock_store()
    from validation_pipeline.schemas.report import FinalReport, DatasetStats, CurationScore, AuditTrail, OutputFiles
    report = FinalReport(
        report_id="abc123", spec_summary="test",
        dataset_stats=DatasetStats(total_images=10, usable=5, recoverable=3, unusable=2, usable_percentage=0.5, flag_breakdown={}),
        curation_score=CurationScore(overall_score=0.5, dimension_scores={}, confidence=0.9, explanation="test"),
        per_image_results=[],
        audit_trail=AuditTrail(spec={}, plan={}, calibration={}, supervision_report={}, timestamp="2026-01-01", llm_model_used="gpt-4o", tool_versions={}),
        output_files=OutputFiles(usable_manifest="x", full_results_json="y"),
    )
    with patch("requests.post", return_value=_mock_response()) as mock_post:
        store.complete_run("abc123", report)
    call_body = mock_post.call_args[1]["json"]
    assert "UPDATE runs" in call_body["query"]
    assert "completed" in call_body["params"]


def test_fail_run():
    store = _mock_store()
    with patch("requests.post", return_value=_mock_response()) as mock_post:
        store.fail_run("abc123", "LLM timeout")
    call_body = mock_post.call_args[1]["json"]
    assert "UPDATE runs" in call_body["query"]
    assert "failed" in call_body["params"]


def test_store_event():
    store = _mock_store()
    from validation_pipeline.events import ModuleStarted
    event = ModuleStarted(module="executor", details="processing")
    with patch("requests.post", return_value=_mock_response()) as mock_post:
        store.store_event("abc123", event)
    call_body = mock_post.call_args[1]["json"]
    assert "INSERT INTO events" in call_body["query"]
    assert "ModuleStarted" in call_body["params"]


def test_store_image_results():
    store = _mock_store()
    from validation_pipeline.schemas.report import ImageReport
    results = [
        ImageReport(image_id="img1", image_path="/img1.jpg", verdict="usable", scores={"blur": 0.9}, flags=[]),
        ImageReport(image_id="img2", image_path="/img2.jpg", verdict="unusable", scores={"blur": 0.2}, flags=["blur"]),
    ]
    # Batch insert sends array of statements
    batch_resp = MagicMock()
    batch_resp.status_code = 200
    batch_resp.json.return_value = [{"rowCount": 1}, {"rowCount": 1}]
    batch_resp.raise_for_status = MagicMock()
    with patch("requests.post", return_value=batch_resp) as mock_post:
        store.store_image_results("abc123", results)
    call_body = mock_post.call_args[1]["json"]
    assert isinstance(call_body, list)
    assert len(call_body) == 2


def test_list_runs():
    store = _mock_store()
    resp = _mock_response(
        rows=[{"id": "id1", "intent": "find horses", "status": "completed"}],
        fields=[{"name": "id"}, {"name": "intent"}, {"name": "status"}],
    )
    with patch("requests.post", return_value=resp):
        runs = store.list_runs(limit=10)
    assert len(runs) == 1
    assert runs[0]["id"] == "id1"


def test_get_run():
    store = _mock_store()
    resp = _mock_response(
        rows=[{"id": "id1", "intent": "find horses", "status": "completed"}],
        fields=[{"name": "id"}, {"name": "intent"}, {"name": "status"}],
    )
    with patch("requests.post", return_value=resp):
        run = store.get_run("id1")
    assert run is not None
    assert run["id"] == "id1"


def test_get_run_not_found():
    store = _mock_store()
    resp = _mock_response(rows=[], fields=[])
    with patch("requests.post", return_value=resp):
        run = store.get_run("nonexistent")
    assert run is None


def test_query_images_by_verdict():
    store = _mock_store()
    resp = _mock_response(
        rows=[{"image_id": "img1", "verdict": "usable"}],
        fields=[{"name": "image_id"}, {"name": "verdict"}],
    )
    with patch("requests.post", return_value=resp) as mock_post:
        images = store.query_images(verdict="usable")
    call_body = mock_post.call_args[1]["json"]
    assert "verdict" in call_body["query"].lower()
    assert len(images) == 1


def test_close_is_noop():
    store = _mock_store()
    store.close()  # Should not raise

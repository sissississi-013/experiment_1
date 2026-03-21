import json
from unittest.mock import MagicMock, patch
from validation_pipeline.persistence.run_store import RunStore

def _mock_store():
    with patch("psycopg2.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn
        store = RunStore("postgresql://fake")
        return store, mock_conn, mock_cursor

def test_run_store_connects():
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value = MagicMock()
        store = RunStore("postgresql://fake")
        mock_connect.assert_called_once_with("postgresql://fake")

def test_create_run():
    store, conn, cursor = _mock_store()
    store.create_run(run_id="abc123", intent="find horses", config_json={"model": "gpt-4o"}, dataset_path="/data/coco")
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "INSERT INTO runs" in sql
    conn.commit.assert_called()

def test_complete_run():
    store, conn, cursor = _mock_store()
    from validation_pipeline.schemas.report import FinalReport, DatasetStats, CurationScore, AuditTrail, OutputFiles
    report = FinalReport(
        report_id="abc123", spec_summary="test",
        dataset_stats=DatasetStats(total_images=10, usable=5, recoverable=3, unusable=2, usable_percentage=0.5, flag_breakdown={}),
        curation_score=CurationScore(overall_score=0.5, dimension_scores={}, confidence=0.9, explanation="test"),
        per_image_results=[],
        audit_trail=AuditTrail(spec={}, plan={}, calibration={}, supervision_report={}, timestamp="2026-01-01", llm_model_used="gpt-4o", tool_versions={}),
        output_files=OutputFiles(usable_manifest="x", full_results_json="y"),
    )
    store.complete_run("abc123", report)
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "UPDATE runs" in sql

def test_fail_run():
    store, conn, cursor = _mock_store()
    store.fail_run("abc123", "LLM timeout")
    cursor.execute.assert_called_once()
    assert "UPDATE runs" in cursor.execute.call_args[0][0]

def test_store_event():
    store, conn, cursor = _mock_store()
    from validation_pipeline.events import ModuleStarted
    event = ModuleStarted(module="executor", details="processing")
    store.store_event("abc123", event)
    cursor.execute.assert_called_once()
    assert "INSERT INTO events" in cursor.execute.call_args[0][0]

def test_store_image_results():
    store, conn, cursor = _mock_store()
    from validation_pipeline.schemas.report import ImageReport
    results = [
        ImageReport(image_id="img1", image_path="/img1.jpg", verdict="usable", scores={"blur": 0.9}, flags=[]),
        ImageReport(image_id="img2", image_path="/img2.jpg", verdict="unusable", scores={"blur": 0.2}, flags=["blur"]),
    ]
    store.store_image_results("abc123", results)
    assert cursor.execute.call_count == 2
    conn.commit.assert_called()

def test_list_runs():
    store, conn, cursor = _mock_store()
    cursor.fetchall.return_value = [("id1", "find horses", "completed", 10, 5, 0.5, "2026-01-01")]
    cursor.description = [("id",), ("intent",), ("status",), ("total_images",), ("usable_count",), ("overall_score",), ("created_at",)]
    runs = store.list_runs(limit=10)
    assert "SELECT" in cursor.execute.call_args[0][0]
    assert "ORDER BY" in cursor.execute.call_args[0][0]

def test_get_run():
    store, conn, cursor = _mock_store()
    cursor.fetchone.return_value = ("id1", "find horses", "completed")
    cursor.description = [("id",), ("intent",), ("status",)]
    result = store.get_run("id1")
    assert "WHERE id" in cursor.execute.call_args[0][0]

def test_query_images_by_verdict():
    store, conn, cursor = _mock_store()
    cursor.fetchall.return_value = []
    cursor.description = [("image_id",), ("verdict",)]
    store.query_images(verdict="usable")
    sql = cursor.execute.call_args[0][0]
    assert "verdict" in sql.lower()

def test_close():
    store, conn, cursor = _mock_store()
    store.close()
    conn.close.assert_called_once()

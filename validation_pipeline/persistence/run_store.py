"""RunStore: Neon Postgres persistence layer using HTTP /sql endpoint.

Uses Neon's serverless HTTP API (port 443) instead of psycopg2 (port 5432),
so it works on networks that block Postgres ports.
"""

import json
from typing import Any

import requests

from validation_pipeline.events import PipelineEvent
from validation_pipeline.schemas.report import FinalReport, ImageReport


class RunStore:
    """Persist pipeline runs, events, and per-image results to Neon Postgres via HTTP."""

    def __init__(self, connection_string: str) -> None:
        # Parse the connection string to extract the host for the /sql endpoint
        # Format: postgresql://user:pass@host/db?params
        self.connection_string = connection_string
        host = connection_string.split("@")[1].split("/")[0]
        self.sql_url = f"https://{host}/sql"
        self.timeout = 15

    def _execute(self, query: str, params: list[Any] | None = None) -> dict:
        """Execute a single SQL query via Neon HTTP endpoint."""
        body: dict[str, Any] = {"query": query}
        if params:
            body["params"] = params
        resp = requests.post(
            self.sql_url,
            headers={
                "Content-Type": "application/json",
                "Neon-Connection-String": self.connection_string,
            },
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _rows_to_dicts(self, result: dict) -> list[dict[str, Any]]:
        """Convert Neon HTTP response to list of dicts."""
        fields = [f["name"] for f in result.get("fields", [])]
        rows = result.get("rows", [])
        if not fields or not rows:
            return []
        # Rows can be dicts (default) or arrays
        if rows and isinstance(rows[0], dict):
            return rows
        return [dict(zip(fields, row)) for row in rows]

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def initialize_schema(self) -> None:
        """Create tables if they do not already exist."""
        for sql in [
            """CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, intent TEXT NOT NULL, dataset_path TEXT,
                dataset_description TEXT, status TEXT NOT NULL DEFAULT 'running',
                total_images INTEGER, usable_count INTEGER, recoverable_count INTEGER,
                unusable_count INTEGER, error_count INTEGER DEFAULT 0, overall_score FLOAT,
                report_json JSONB, config_json JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(), completed_at TIMESTAMPTZ
            )""",
            """CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY, run_id TEXT REFERENCES runs(id),
                event_type TEXT NOT NULL, module TEXT NOT NULL, payload JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS image_results (
                id SERIAL PRIMARY KEY, run_id TEXT REFERENCES runs(id),
                image_id TEXT NOT NULL, image_path TEXT NOT NULL, verdict TEXT NOT NULL,
                scores JSONB DEFAULT '{}', errors JSONB DEFAULT '[]', flags JSONB DEFAULT '[]',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )""",
            "CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_image_results_run_id ON image_results(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_image_results_verdict ON image_results(verdict)",
        ]:
            self._execute(sql)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(self, run_id: str, intent: str, config_json: dict,
                   dataset_path: str | None = None,
                   dataset_description: str | None = None) -> str:
        self._execute(
            "INSERT INTO runs (id, intent, dataset_path, dataset_description, status, config_json) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            [run_id, intent, dataset_path, dataset_description, "running", json.dumps(config_json)],
        )
        return run_id

    def complete_run(self, run_id: str, report: FinalReport) -> None:
        self._execute(
            "UPDATE runs SET status = $1, total_images = $2, usable_count = $3, "
            "recoverable_count = $4, unusable_count = $5, error_count = $6, "
            "overall_score = $7, report_json = $8, completed_at = NOW() WHERE id = $9",
            [
                "completed",
                report.dataset_stats.total_images,
                report.dataset_stats.usable,
                report.dataset_stats.recoverable,
                report.dataset_stats.unusable,
                report.dataset_stats.error_count,
                report.curation_score.overall_score,
                report.model_dump_json(),
                run_id,
            ],
        )

    def fail_run(self, run_id: str, error: str) -> None:
        self._execute(
            "UPDATE runs SET status = $1, completed_at = NOW(), "
            "report_json = $2 WHERE id = $3",
            ["failed", json.dumps({"error": error}), run_id],
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def store_event(self, run_id: str, event: PipelineEvent) -> None:
        self._execute(
            "INSERT INTO events (run_id, event_type, module, payload) "
            "VALUES ($1, $2, $3, $4)",
            [run_id, type(event).__name__, event.module, event.model_dump_json()],
        )

    # ------------------------------------------------------------------
    # Image results
    # ------------------------------------------------------------------

    def store_image_results(self, run_id: str, results: list[ImageReport]) -> None:
        for img in results:
            self._execute(
                "INSERT INTO image_results (run_id, image_id, image_path, verdict, scores, errors, flags) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                [
                    run_id, img.image_id, img.image_path, img.verdict,
                    json.dumps(img.scores), json.dumps([]),
                    json.dumps(img.flags),
                ],
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        result = self._execute("SELECT * FROM runs WHERE id = $1", [run_id])
        rows = self._rows_to_dicts(result)
        return rows[0] if rows else None

    def list_runs(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        result = self._execute(
            "SELECT id, intent, status, total_images, usable_count, overall_score, created_at "
            "FROM runs ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            [limit, offset],
        )
        return self._rows_to_dicts(result)

    def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        result = self._execute(
            "SELECT * FROM events WHERE run_id = $1 ORDER BY created_at", [run_id],
        )
        return self._rows_to_dicts(result)

    def get_run_images(self, run_id: str, verdict: str | None = None) -> list[dict[str, Any]]:
        if verdict:
            result = self._execute(
                "SELECT * FROM image_results WHERE run_id = $1 AND verdict = $2",
                [run_id, verdict],
            )
        else:
            result = self._execute(
                "SELECT * FROM image_results WHERE run_id = $1", [run_id],
            )
        return self._rows_to_dicts(result)

    def query_images(self, verdict: str | None = None, min_score: float | None = None,
                     dimension: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        conditions = []
        params = []
        idx = 1
        if verdict:
            conditions.append(f"verdict = ${idx}")
            params.append(verdict)
            idx += 1
        if min_score is not None and dimension:
            conditions.append(f"(scores->>'{dimension}')::float >= ${idx}")
            params.append(min_score)
            idx += 1
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        result = self._execute(
            f"SELECT * FROM image_results {where} ORDER BY created_at DESC LIMIT ${idx}",
            params,
        )
        return self._rows_to_dicts(result)

    def close(self) -> None:
        """No-op for HTTP-based store (no persistent connection)."""
        pass

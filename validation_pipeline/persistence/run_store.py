"""RunStore: Neon Postgres persistence layer for pipeline runs, events, and image results."""

import json
from typing import Any

import psycopg2

from validation_pipeline.events import PipelineEvent
from validation_pipeline.schemas.report import FinalReport, ImageReport


class RunStore:
    """Persist pipeline runs, events, and per-image results to a Postgres database."""

    def __init__(self, dsn: str) -> None:
        self.conn = psycopg2.connect(dsn)

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def initialize_schema(self) -> None:
        """Create tables if they do not already exist."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    intent TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    config_json JSONB,
                    dataset_path TEXT,
                    report_json JSONB,
                    error_message TEXT,
                    total_images INTEGER,
                    usable_count INTEGER,
                    overall_score FLOAT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    module TEXT,
                    payload JSONB,
                    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS image_results (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    image_id TEXT NOT NULL,
                    image_path TEXT,
                    verdict TEXT,
                    scores JSONB,
                    flags JSONB,
                    recovery_suggestion TEXT,
                    explanation TEXT
                )
            """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        intent: str,
        config_json: dict,
        dataset_path: str,
    ) -> None:
        """Insert a new run row with status 'running'."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (id, intent, status, config_json, dataset_path)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (run_id, intent, "running", json.dumps(config_json), dataset_path),
            )
        self.conn.commit()

    def complete_run(self, run_id: str, report: FinalReport) -> None:
        """Mark a run as completed and store the final report."""
        stats = report.dataset_stats
        score = report.curation_score.overall_score
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                SET status = %s,
                    report_json = %s,
                    total_images = %s,
                    usable_count = %s,
                    overall_score = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (
                    "completed",
                    report.model_dump_json(),
                    stats.total_images,
                    stats.usable,
                    score,
                    run_id,
                ),
            )
        self.conn.commit()

    def fail_run(self, run_id: str, error_message: str) -> None:
        """Mark a run as failed and record the error message."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                SET status = %s,
                    error_message = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                ("failed", error_message, run_id),
            )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def store_event(self, run_id: str, event: PipelineEvent) -> None:
        """Persist a pipeline event row."""
        event_type = type(event).__name__
        payload = event.model_dump_json()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO events (run_id, event_type, module, payload)
                VALUES (%s, %s, %s, %s)
                """,
                (run_id, event_type, event.module, payload),
            )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Image results
    # ------------------------------------------------------------------

    def store_image_results(self, run_id: str, results: list[ImageReport]) -> None:
        """Bulk-insert per-image results."""
        with self.conn.cursor() as cur:
            for result in results:
                cur.execute(
                    """
                    INSERT INTO image_results
                        (run_id, image_id, image_path, verdict, scores, flags, recovery_suggestion, explanation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        result.image_id,
                        result.image_path,
                        result.verdict,
                        json.dumps(result.scores),
                        json.dumps(result.flags),
                        result.recovery_suggestion,
                        result.explanation,
                    ),
                )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _rows_to_dicts(self, cursor) -> list[dict[str, Any]]:
        if cursor.description is None:
            return []
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in (cursor.fetchall())]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a single run row as a dict, or None if not found."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM runs WHERE id = %s",
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            if cur.description is None:
                return None
            columns = [col[0] for col in cur.description]
            return dict(zip(columns, row))

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent runs ordered by creation time descending."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            if not rows or cur.description is None:
                return []
            columns = [col[0] for col in cur.description]
            return [dict(zip(columns, row)) for row in rows]

    def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        """Return all events for a run ordered by occurrence time."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM events WHERE run_id = %s ORDER BY occurred_at ASC",
                (run_id,),
            )
            rows = cur.fetchall()
            if not rows or cur.description is None:
                return []
            columns = [col[0] for col in cur.description]
            return [dict(zip(columns, row)) for row in rows]

    def get_run_images(self, run_id: str) -> list[dict[str, Any]]:
        """Return all image results for a run."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM image_results WHERE run_id = %s",
                (run_id,),
            )
            rows = cur.fetchall()
            if not rows or cur.description is None:
                return []
            columns = [col[0] for col in cur.description]
            return [dict(zip(columns, row)) for row in rows]

    def query_images(
        self,
        run_id: str | None = None,
        verdict: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query image results with optional filters on run_id and verdict."""
        conditions: list[str] = []
        params: list[Any] = []

        if run_id is not None:
            conditions.append("run_id = %s")
            params.append(run_id)
        if verdict is not None:
            conditions.append("verdict = %s")
            params.append(verdict)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        sql = f"SELECT * FROM image_results {where_clause} LIMIT %s"

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            if not rows or cur.description is None:
                return []
            columns = [col[0] for col in cur.description]
            return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self.conn.close()

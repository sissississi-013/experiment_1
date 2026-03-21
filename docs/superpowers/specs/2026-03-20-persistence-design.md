# Persistence & Run History — Backend Polish Sprint C

**Date**: 2026-03-20
**Status**: Draft
**Sprint**: C (Persistence) — third of three backend polish sprints (B → A → C)

## Overview

Persist pipeline runs, events, and per-image results to Neon (serverless Postgres) so run history is queryable, reproducible, and ready for the frontend. Persistence plugs in as an EventBus subscriber — zero changes to pipeline code.

## Why

Currently everything is ephemeral — the report goes to a JSON file that gets overwritten on the next run. There's no way to compare runs, query which images keep failing, or show historical data in a frontend. For Apple-grade engineering, run history must be durable, queryable, and auditable.

---

## 1. Database Schema

**Three tables** on Neon Postgres:

```sql
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    intent TEXT NOT NULL,
    dataset_path TEXT,
    dataset_description TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    total_images INTEGER,
    usable_count INTEGER,
    recoverable_count INTEGER,
    unusable_count INTEGER,
    error_count INTEGER DEFAULT 0,
    overall_score FLOAT,
    report_json JSONB,
    config_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    run_id TEXT REFERENCES runs(id),
    event_type TEXT NOT NULL,
    module TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE image_results (
    id SERIAL PRIMARY KEY,
    run_id TEXT REFERENCES runs(id),
    image_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    verdict TEXT NOT NULL,
    scores JSONB DEFAULT '{}',
    errors JSONB DEFAULT '[]',
    flags JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_run_id ON events(run_id);
CREATE INDEX idx_image_results_run_id ON image_results(run_id);
CREATE INDEX idx_image_results_verdict ON image_results(verdict);
```

**Status values for runs:** `"running"`, `"completed"`, `"failed"`

---

## 2. RunStore

**New file**: `validation_pipeline/persistence/run_store.py`

A single class that handles all database operations. Only code that talks to Neon.

```python
class RunStore:
    def __init__(self, connection_string: str):
        """Connect to Neon Postgres."""

    def initialize_schema(self):
        """Create tables if they don't exist (idempotent)."""

    # Run lifecycle
    def create_run(self, run_id: str, intent: str, config_json: dict,
                   dataset_path: str | None = None,
                   dataset_description: str | None = None) -> str:
        """Insert a new run with status='running'. Returns run_id."""

    def complete_run(self, run_id: str, report: FinalReport):
        """Update run with final stats, report JSON, status='completed'."""

    def fail_run(self, run_id: str, error: str):
        """Update run with status='failed' and error info."""

    # Events
    def store_event(self, run_id: str, event: PipelineEvent):
        """Insert a single event row."""

    # Image results
    def store_image_results(self, run_id: str, results: list[ImageReport]):
        """Batch insert image results for a run."""

    # Queries
    def get_run(self, run_id: str) -> dict | None:
        """Get a single run by ID."""

    def list_runs(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """List recent runs, newest first."""

    def get_run_events(self, run_id: str) -> list[dict]:
        """Get all events for a run, ordered by timestamp."""

    def get_run_images(self, run_id: str, verdict: str | None = None) -> list[dict]:
        """Get image results for a run, optionally filtered by verdict."""

    def query_images(self, verdict: str | None = None,
                     min_score: float | None = None,
                     dimension: str | None = None,
                     limit: int = 100) -> list[dict]:
        """Query images across all runs."""

    def close(self):
        """Close database connection."""
```

Uses `psycopg2` for direct SQL. No ORM — keeps queries transparent and debuggable.

---

## 3. PersistenceSubscriber

**New file**: `validation_pipeline/persistence/subscriber.py`

An EventBus subscriber that writes events to the database as they arrive:

```python
class PersistenceSubscriber:
    def __init__(self, store: RunStore, run_id: str):
        self.store = store
        self.run_id = run_id

    def __call__(self, event: PipelineEvent):
        """Called by EventBus for every event. Never raises."""
        try:
            self.store.store_event(self.run_id, event)
        except Exception:
            pass  # DB write failure never kills the pipeline
```

Errors are swallowed — persistence is best-effort. The pipeline must never fail because the database is down.

---

## 4. Integration

**Modify**: `run_pipeline.py`

```python
store = RunStore(os.environ.get("NEON_DATABASE_URL", ""))
store.initialize_schema()
run_id = store.create_run(intent=intent, config_json=config.model_dump(), ...)

persistence = PersistenceSubscriber(store, run_id)
bus.subscribe_all(cli_subscriber)
bus.subscribe_all(persistence)

pipeline = ValidationPipeline(config, event_bus=bus)

try:
    report = pipeline.run(user_input, auto_approve=True)
    store.complete_run(run_id, report)
    store.store_image_results(run_id, report.per_image_results)
except Exception as e:
    store.fail_run(run_id, str(e))
    raise
finally:
    store.close()
```

**Persistence is optional**: If `NEON_DATABASE_URL` is not set, skip persistence entirely. The pipeline works the same as before — events go to CLI subscriber only.

**Connection string** from `NEON_DATABASE_URL` env var in `.env`.

---

## 5. New Dependency

Add `psycopg2-binary>=2.9` to `pyproject.toml` dependencies.

---

## 6. Test Strategy

- `tests/test_persistence/test_run_store.py` — Unit tests with a real Neon test database (or mocked). Tests: create_run, complete_run, fail_run, store_event, store_image_results, list_runs, get_run, query_images.
- `tests/test_persistence/test_subscriber.py` — PersistenceSubscriber swallows errors, correctly passes events to store.
- Tests marked `@pytest.mark.integration` for the ones that need a real database.
- Unit tests mock the database connection.

---

## 7. File Structure

### New Files
```
validation_pipeline/persistence/__init__.py
validation_pipeline/persistence/run_store.py      # RunStore class
validation_pipeline/persistence/subscriber.py      # PersistenceSubscriber
tests/test_persistence/__init__.py
tests/test_persistence/test_run_store.py
tests/test_persistence/test_subscriber.py
```

### Modified Files
```
run_pipeline.py                  # Wire up persistence
pyproject.toml                   # Add psycopg2-binary
.env                             # Add NEON_DATABASE_URL
```

---

## 8. What's Explicitly Out of Scope

- No REST API for querying runs (that's frontend work)
- No migration system (schema is simple enough for CREATE IF NOT EXISTS)
- No connection pooling (single-user, single connection is fine)
- No async database operations
- Pipeline code unchanged — persistence is purely a subscriber

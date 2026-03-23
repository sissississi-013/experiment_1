# Persistence & Run History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist pipeline runs, events, and per-image results to Neon Postgres so run history is queryable and ready for the frontend.

**Architecture:** A `RunStore` class handles all database operations via `psycopg2`. A `PersistenceSubscriber` plugs into the existing EventBus as a subscriber — zero changes to pipeline code. Tables: `runs`, `events`, `image_results`.

**Tech Stack:** Python 3.14, psycopg2-binary, Neon Postgres

**Spec:** `docs/superpowers/specs/2026-03-20-persistence-design.md`

**Neon connection string:** In `.env` as `NEON_DATABASE_URL`

---

### Task 1: Add psycopg2 dependency + create package

**Files:**
- Modify: `pyproject.toml`
- Create: `validation_pipeline/persistence/__init__.py`
- Create: `tests/test_persistence/__init__.py`

- [ ] **Step 1: Add psycopg2-binary to pyproject.toml**

Add `"psycopg2-binary>=2.9"` to the main dependencies list in `pyproject.toml`.

- [ ] **Step 2: Create package directories**

Create empty `__init__.py` files:
- `validation_pipeline/persistence/__init__.py`
- `tests/test_persistence/__init__.py`

- [ ] **Step 3: Install**

Run: `cd /Users/sissi/Desktop/validation-pipeline && pip3 install -e ".[dev]" --break-system-packages`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml validation_pipeline/persistence/__init__.py tests/test_persistence/__init__.py
git commit -m "feat: add psycopg2-binary dependency and persistence package"
```

---

### Task 2: RunStore — schema + run lifecycle

**Files:**
- Create: `validation_pipeline/persistence/run_store.py`
- Test: `tests/test_persistence/test_run_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_persistence/test_run_store.py
import json
from unittest.mock import MagicMock, patch, call
from validation_pipeline.persistence.run_store import RunStore


def _mock_store():
    """Create a RunStore with mocked database connection."""
    with patch("psycopg2.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
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
    store.create_run(
        run_id="abc123",
        intent="find horses",
        config_json={"model": "gpt-4o"},
        dataset_path="/data/coco",
    )
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "INSERT INTO runs" in sql
    conn.commit.assert_called()


def test_complete_run():
    store, conn, cursor = _mock_store()
    from validation_pipeline.schemas.report import (
        FinalReport, DatasetStats, CurationScore, AuditTrail, OutputFiles,
    )
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
    assert "completed" in str(cursor.execute.call_args[0][1]).lower() or "status" in sql.lower()


def test_fail_run():
    store, conn, cursor = _mock_store()
    store.fail_run("abc123", "LLM timeout")
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "UPDATE runs" in sql


def test_store_event():
    store, conn, cursor = _mock_store()
    from validation_pipeline.events import ModuleStarted
    event = ModuleStarted(module="executor", details="processing")
    store.store_event("abc123", event)
    cursor.execute.assert_called_once()
    sql = cursor.execute.call_args[0][0]
    assert "INSERT INTO events" in sql


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
    cursor.fetchall.return_value = [
        ("id1", "find horses", "completed", 10, 5, 0.5, "2026-01-01"),
    ]
    cursor.description = [("id",), ("intent",), ("status",), ("total_images",), ("usable_count",), ("overall_score",), ("created_at",)]
    runs = store.list_runs(limit=10)
    cursor.execute.assert_called_once()
    assert "SELECT" in cursor.execute.call_args[0][0]
    assert "ORDER BY" in cursor.execute.call_args[0][0]


def test_get_run():
    store, conn, cursor = _mock_store()
    cursor.fetchone.return_value = ("id1", "find horses", "completed")
    cursor.description = [("id",), ("intent",), ("status",)]
    result = store.get_run("id1")
    assert "SELECT" in cursor.execute.call_args[0][0]
    assert "WHERE id" in cursor.execute.call_args[0][0]


def test_query_images_by_verdict():
    store, conn, cursor = _mock_store()
    cursor.fetchall.return_value = []
    cursor.description = [("image_id",), ("verdict",)]
    store.query_images(verdict="usable")
    sql = cursor.execute.call_args[0][0]
    assert "SELECT" in sql
    assert "verdict" in sql.lower()


def test_close():
    store, conn, cursor = _mock_store()
    store.close()
    conn.close.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_persistence/test_run_store.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement RunStore**

```python
# validation_pipeline/persistence/run_store.py
import json
import psycopg2
from validation_pipeline.events import PipelineEvent
from validation_pipeline.schemas.report import FinalReport, ImageReport


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
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

CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    run_id TEXT REFERENCES runs(id),
    event_type TEXT NOT NULL,
    module TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS image_results (
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

CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_image_results_run_id ON image_results(run_id);
CREATE INDEX IF NOT EXISTS idx_image_results_verdict ON image_results(verdict);
"""


class RunStore:
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)

    def initialize_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        self.conn.commit()

    def create_run(self, run_id: str, intent: str, config_json: dict,
                   dataset_path: str | None = None,
                   dataset_description: str | None = None) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO runs (id, intent, dataset_path, dataset_description, status, config_json) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (run_id, intent, dataset_path, dataset_description, "running", json.dumps(config_json)),
            )
        self.conn.commit()
        return run_id

    def complete_run(self, run_id: str, report: FinalReport):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET status = %s, total_images = %s, usable_count = %s, "
                "recoverable_count = %s, unusable_count = %s, error_count = %s, "
                "overall_score = %s, report_json = %s, completed_at = NOW() WHERE id = %s",
                (
                    "completed",
                    report.dataset_stats.total_images,
                    report.dataset_stats.usable,
                    report.dataset_stats.recoverable,
                    report.dataset_stats.unusable,
                    report.dataset_stats.error_count,
                    report.curation_score.overall_score,
                    report.model_dump_json(),
                    run_id,
                ),
            )
        self.conn.commit()

    def fail_run(self, run_id: str, error: str):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET status = %s, completed_at = NOW(), "
                "report_json = %s WHERE id = %s",
                ("failed", json.dumps({"error": error}), run_id),
            )
        self.conn.commit()

    def store_event(self, run_id: str, event: PipelineEvent):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (run_id, event_type, module, payload) "
                "VALUES (%s, %s, %s, %s)",
                (run_id, type(event).__name__, event.module, event.model_dump_json()),
            )
        self.conn.commit()

    def store_image_results(self, run_id: str, results: list[ImageReport]):
        with self.conn.cursor() as cur:
            for img in results:
                cur.execute(
                    "INSERT INTO image_results (run_id, image_id, image_path, verdict, scores, errors, flags) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        run_id, img.image_id, img.image_path, img.verdict,
                        json.dumps(img.scores), json.dumps([]),
                        json.dumps(img.flags),
                    ),
                )
        self.conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))

    def list_runs(self, limit: int = 20, offset: int = 0) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, intent, status, total_images, usable_count, overall_score, created_at "
                "FROM runs ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, offset),
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_run_events(self, run_id: str) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM events WHERE run_id = %s ORDER BY created_at",
                (run_id,),
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_run_images(self, run_id: str, verdict: str | None = None) -> list[dict]:
        with self.conn.cursor() as cur:
            if verdict:
                cur.execute(
                    "SELECT * FROM image_results WHERE run_id = %s AND verdict = %s",
                    (run_id, verdict),
                )
            else:
                cur.execute("SELECT * FROM image_results WHERE run_id = %s", (run_id,))
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def query_images(self, verdict: str | None = None, min_score: float | None = None,
                     dimension: str | None = None, limit: int = 100) -> list[dict]:
        conditions = []
        params = []
        if verdict:
            conditions.append("verdict = %s")
            params.append(verdict)
        if min_score is not None and dimension:
            conditions.append("(scores->>%s)::float >= %s")
            params.append(dimension)
            params.append(min_score)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM image_results {where} ORDER BY created_at DESC LIMIT %s",
                params,
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self):
        self.conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_persistence/test_run_store.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add validation_pipeline/persistence/run_store.py tests/test_persistence/test_run_store.py
git commit -m "feat: add RunStore for Neon Postgres persistence"
```

---

### Task 3: PersistenceSubscriber

**Files:**
- Create: `validation_pipeline/persistence/subscriber.py`
- Test: `tests/test_persistence/test_subscriber.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_persistence/test_subscriber.py
from unittest.mock import MagicMock
from validation_pipeline.persistence.subscriber import PersistenceSubscriber
from validation_pipeline.events import ModuleStarted, ImageVerdict


def test_subscriber_calls_store_event():
    store = MagicMock()
    sub = PersistenceSubscriber(store, "run123")
    event = ModuleStarted(module="test")
    sub(event)
    store.store_event.assert_called_once_with("run123", event)


def test_subscriber_swallows_errors():
    store = MagicMock()
    store.store_event.side_effect = Exception("DB down")
    sub = PersistenceSubscriber(store, "run123")
    # Should not raise
    sub(ModuleStarted(module="test"))


def test_subscriber_passes_run_id():
    store = MagicMock()
    sub = PersistenceSubscriber(store, "my-run-id")
    sub(ImageVerdict(module="executor", image_id="x", image_path="/x.jpg", verdict="usable"))
    assert store.store_event.call_args[0][0] == "my-run-id"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_persistence/test_subscriber.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement PersistenceSubscriber**

```python
# validation_pipeline/persistence/subscriber.py
from validation_pipeline.events import PipelineEvent
from validation_pipeline.persistence.run_store import RunStore


class PersistenceSubscriber:
    def __init__(self, store: RunStore, run_id: str):
        self.store = store
        self.run_id = run_id

    def __call__(self, event: PipelineEvent):
        try:
            self.store.store_event(self.run_id, event)
        except Exception:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_persistence/test_subscriber.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add validation_pipeline/persistence/subscriber.py tests/test_persistence/test_subscriber.py
git commit -m "feat: add PersistenceSubscriber for EventBus"
```

---

### Task 4: Initialize schema on Neon

**Files:** None (SQL execution)

- [ ] **Step 1: Create tables on Neon**

Run:
```bash
cd /Users/sissi/Desktop/validation-pipeline && python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
from validation_pipeline.persistence.run_store import RunStore
store = RunStore(os.environ['NEON_DATABASE_URL'])
store.initialize_schema()
store.close()
print('Schema created successfully')
"
```
Expected: `Schema created successfully`

- [ ] **Step 2: Verify tables exist**

Run:
```bash
cd /Users/sissi/Desktop/validation-pipeline && python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
from validation_pipeline.persistence.run_store import RunStore
store = RunStore(os.environ['NEON_DATABASE_URL'])
with store.conn.cursor() as cur:
    cur.execute(\"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'\")
    tables = [row[0] for row in cur.fetchall()]
print(f'Tables: {tables}')
store.close()
"
```
Expected: `Tables: ['runs', 'events', 'image_results']`

---

### Task 5: Wire persistence into run_pipeline.py

**Files:**
- Modify: `run_pipeline.py`

- [ ] **Step 1: Update run_pipeline.py**

Add imports after existing imports:
```python
from validation_pipeline.persistence.run_store import RunStore
from validation_pipeline.persistence.subscriber import PersistenceSubscriber
```

In the `run()` function, after creating the EventBus and before creating the pipeline, add persistence wiring. The persistence should be optional — only activate if `NEON_DATABASE_URL` is set:

```python
def run(intent, dataset_path=None, dataset_description=None):
    bus = EventBus()
    bus.subscribe_all(cli_subscriber)

    # Optional persistence
    store = None
    run_id = None
    db_url = os.environ.get("NEON_DATABASE_URL", "")
    if db_url:
        try:
            store = RunStore(db_url)
            import uuid
            run_id = str(uuid.uuid4())[:8]
            store.create_run(
                run_id=run_id,
                intent=intent,
                config_json={"openai_api_key": "***", "llm_model": "gpt-4o"},
                dataset_path=dataset_path,
                dataset_description=dataset_description,
            )
            persistence = PersistenceSubscriber(store, run_id)
            bus.subscribe_all(persistence)
            print(f"  [persistence] Connected to Neon (run_id={run_id})")
        except Exception as e:
            print(f"  [persistence] WARNING: Could not connect to database: {e}")
            store = None

    config = PipelineConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    pipeline = ValidationPipeline(config, event_bus=bus)

    # ... existing user_input creation ...

    try:
        report = pipeline.run(user_input, auto_approve=True)
        if store and run_id:
            store.complete_run(run_id, report)
            store.store_image_results(run_id, report.per_image_results)
            print(f"  [persistence] Run saved to database")
    except Exception as e:
        if store and run_id:
            store.fail_run(run_id, str(e))
        raise
    finally:
        if store:
            store.close()

    # ... existing report printing ...
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m "not integration"`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add run_pipeline.py
git commit -m "feat: wire Neon persistence into CLI runner (optional)"
```

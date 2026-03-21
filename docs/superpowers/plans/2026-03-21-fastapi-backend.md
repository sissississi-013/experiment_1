# FastAPI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend that exposes the validation pipeline over REST + WebSocket, enabling the Next.js frontend to start runs, stream live progress, browse history, and export curated datasets.

**Architecture:** FastAPI wraps the existing `ValidationPipeline`, `EventBus`, and `RunStore`. Pipeline runs execute as background tasks. WebSocket bridges the EventBus to connected clients. Export generates ZIP files on-the-fly from image results.

**Tech Stack:** Python 3.14, FastAPI, uvicorn, python-multipart, websockets

**Spec:** `docs/superpowers/specs/2026-03-21-frontend-design.md` (Section 1: Architecture)

---

### Task 1: FastAPI scaffold + dependencies

**Files:**
- Create: `api/__init__.py`
- Create: `api/server.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Add to main dependencies: `"fastapi>=0.110"`, `"uvicorn[standard]>=0.27"`, `"python-multipart>=0.0.6"`

- [ ] **Step 2: Create api package**

Create `api/__init__.py` (empty).

- [ ] **Step 3: Create FastAPI server**

```python
# api/server.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from validation_pipeline.config import PipelineConfig
from validation_pipeline.persistence.run_store import RunStore


def get_config() -> PipelineConfig:
    return PipelineConfig(openai_api_key=os.environ.get("OPENAI_API_KEY", ""))


def get_store() -> RunStore | None:
    db_url = os.environ.get("NEON_DATABASE_URL", "")
    if db_url:
        return RunStore(db_url)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config = get_config()
    app.state.store = get_store()
    yield


app = FastAPI(title="Validation Pipeline API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Install and test**

Run: `cd /Users/sissi/Desktop/validation-pipeline && pip3 install -e ".[dev]" --break-system-packages`
Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -c "from api.server import app; print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add api/ pyproject.toml
git commit -m "feat: FastAPI scaffold with health endpoint and CORS"
```

---

### Task 2: Runs endpoints — list + get + create

**Files:**
- Create: `api/routes/__init__.py`
- Create: `api/routes/runs.py`
- Modify: `api/server.py` (register router)
- Test: `tests/test_api/test_runs.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api/__init__.py (empty)

# tests/test_api/test_runs.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
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
    assert resp.json()["status"] == "ok"


def test_list_runs(client):
    app.state.store.list_runs.return_value = [
        {"id": "abc", "intent": "test", "status": "completed", "total_images": 10, "usable_count": 5, "overall_score": 0.5, "created_at": "2026-01-01"},
    ]
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == "abc"


def test_get_run(client):
    app.state.store.get_run.return_value = {"id": "abc", "intent": "test", "status": "completed"}
    resp = client.get("/api/runs/abc")
    assert resp.status_code == 200
    assert resp.json()["id"] == "abc"


def test_get_run_not_found(client):
    app.state.store.get_run.return_value = None
    resp = client.get("/api/runs/nonexistent")
    assert resp.status_code == 404


def test_create_run(client):
    app.state.store.create_run.return_value = "new-run-id"

    with patch("api.routes.runs.run_pipeline_background") as mock_bg:
        resp = client.post("/api/runs", json={
            "intent": "find horses",
            "dataset_description": "10 horse images from COCO",
        })

    assert resp.status_code == 201
    data = resp.json()
    assert "run_id" in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_api/ -v`
Expected: FAIL

- [ ] **Step 3: Implement runs router**

```python
# api/routes/__init__.py (empty)

# api/routes/runs.py
import uuid
import threading
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from validation_pipeline.config import PipelineConfig
from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.pipeline import ValidationPipeline
from validation_pipeline.event_bus import EventBus
from validation_pipeline.persistence.subscriber import PersistenceSubscriber


router = APIRouter(prefix="/api/runs", tags=["runs"])


class CreateRunRequest(BaseModel):
    intent: str
    dataset_path: str | None = None
    dataset_description: str | None = None
    max_images: int | None = None


class CreateRunResponse(BaseModel):
    run_id: str
    status: str = "running"


def run_pipeline_background(run_id: str, request: CreateRunRequest, config: PipelineConfig, store):
    """Run the pipeline in a background thread."""
    bus = EventBus()
    if store:
        persistence = PersistenceSubscriber(store, run_id)
        bus.subscribe_all(persistence)

    # Store active bus for WebSocket streaming
    _active_runs[run_id] = bus

    pipeline = ValidationPipeline(config, event_bus=bus)
    user_input = UserInput(
        intent=request.intent,
        dataset_path=request.dataset_path,
        dataset_description=request.dataset_description or request.intent,
    )

    try:
        report = pipeline.run(user_input, auto_approve=True)
        if store:
            store.complete_run(run_id, report)
            store.store_image_results(run_id, report.per_image_results)
    except Exception as e:
        if store:
            store.fail_run(run_id, str(e))
    finally:
        _active_runs.pop(run_id, None)


# Track active runs for WebSocket streaming
_active_runs: dict[str, EventBus] = {}


def get_active_bus(run_id: str) -> EventBus | None:
    return _active_runs.get(run_id)


@router.get("")
async def list_runs(request: Request, limit: int = 20, offset: int = 0):
    store = request.app.state.store
    if not store:
        return []
    return store.list_runs(limit=limit, offset=offset)


@router.get("/{run_id}")
async def get_run(run_id: str, request: Request):
    store = request.app.state.store
    if not store:
        raise HTTPException(404, "No database configured")
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return run


@router.post("", status_code=201)
async def create_run(body: CreateRunRequest, request: Request):
    config = request.app.state.config
    store = request.app.state.store
    run_id = str(uuid.uuid4())[:8]

    if store:
        store.create_run(
            run_id=run_id,
            intent=body.intent,
            config_json=config.model_dump(),
            dataset_path=body.dataset_path,
            dataset_description=body.dataset_description,
        )

    thread = threading.Thread(
        target=run_pipeline_background,
        args=(run_id, body, config, store),
        daemon=True,
    )
    thread.start()

    return CreateRunResponse(run_id=run_id)
```

- [ ] **Step 4: Register router in server.py**

Add to `api/server.py`:
```python
from api.routes.runs import router as runs_router
app.include_router(runs_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_api/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add api/routes/ tests/test_api/ api/server.py
git commit -m "feat: add runs REST endpoints (list, get, create with background execution)"
```

---

### Task 3: Images endpoint

**Files:**
- Create: `api/routes/images.py`
- Modify: `api/server.py` (register router)
- Test: `tests/test_api/test_images.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api/test_images.py
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from api.server import app
import pytest


@pytest.fixture
def client():
    app.state.config = MagicMock()
    app.state.store = MagicMock()
    return TestClient(app)


def test_get_run_images(client):
    app.state.store.get_run_images.return_value = [
        {"image_id": "img1", "verdict": "usable", "scores": {"blur": 0.9}},
    ]
    resp = client.get("/api/runs/abc/images")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_run_images_filtered(client):
    app.state.store.get_run_images.return_value = []
    resp = client.get("/api/runs/abc/images?verdict=usable")
    assert resp.status_code == 200
    app.state.store.get_run_images.assert_called_with("abc", verdict="usable")


def test_query_images_across_runs(client):
    app.state.store.query_images.return_value = [
        {"image_id": "img1", "verdict": "usable"},
    ]
    resp = client.get("/api/images?verdict=usable&limit=10")
    assert resp.status_code == 200
```

- [ ] **Step 2: Implement images router**

```python
# api/routes/images.py
from fastapi import APIRouter, Request

router = APIRouter(tags=["images"])


@router.get("/api/runs/{run_id}/images")
async def get_run_images(run_id: str, request: Request, verdict: str | None = None):
    store = request.app.state.store
    if not store:
        return []
    return store.get_run_images(run_id, verdict=verdict)


@router.get("/api/images")
async def query_images(
    request: Request,
    verdict: str | None = None,
    min_score: float | None = None,
    dimension: str | None = None,
    limit: int = 100,
):
    store = request.app.state.store
    if not store:
        return []
    return store.query_images(verdict=verdict, min_score=min_score, dimension=dimension, limit=limit)
```

- [ ] **Step 3: Register router, run tests, commit**

Add to server.py: `from api.routes.images import router as images_router` and `app.include_router(images_router)`.

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest tests/test_api/ -v`

```bash
git add api/routes/images.py tests/test_api/test_images.py api/server.py
git commit -m "feat: add images REST endpoints (per-run and cross-run queries)"
```

---

### Task 4: Export endpoint — ZIP download

**Files:**
- Create: `api/routes/export.py`
- Modify: `api/server.py` (register router)
- Test: `tests/test_api/test_export.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api/test_export.py
import json
import zipfile
import io
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from api.server import app
import pytest


@pytest.fixture
def client():
    app.state.config = MagicMock()
    app.state.store = MagicMock()
    return TestClient(app)


def test_export_returns_zip(client, tmp_path):
    # Create a fake image file
    img_path = tmp_path / "img1.jpg"
    img_path.write_bytes(b"\xff\xd8fake_jpg")

    app.state.store.get_run.return_value = {"id": "abc", "status": "completed"}
    app.state.store.get_run_images.return_value = [
        {"image_id": "img1", "image_path": str(img_path), "verdict": "usable", "scores": {"blur": 0.9}, "flags": []},
    ]

    resp = client.get("/api/runs/abc/export?filter=usable")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    # Verify ZIP contents
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    names = z.namelist()
    assert "manifest.json" in names
    assert any(n.endswith(".jpg") for n in names)

    manifest = json.loads(z.read("manifest.json"))
    assert len(manifest["images"]) == 1
    assert manifest["images"][0]["verdict"] == "usable"


def test_export_run_not_found(client):
    app.state.store.get_run.return_value = None
    resp = client.get("/api/runs/nonexistent/export")
    assert resp.status_code == 404
```

- [ ] **Step 2: Implement export router**

```python
# api/routes/export.py
import io
import json
import zipfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["export"])


@router.get("/api/runs/{run_id}/export")
async def export_run(
    run_id: str,
    request: Request,
    filter: str = "usable",
    include_manifest: bool = True,
    include_report: bool = False,
):
    store = request.app.state.store
    if not store:
        raise HTTPException(500, "No database configured")

    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")

    # Determine verdict filter
    if filter == "usable":
        verdicts = ["usable"]
    elif filter == "usable+recoverable":
        verdicts = ["usable", "recoverable"]
    else:
        verdicts = None  # all

    # Get images
    images = store.get_run_images(run_id)
    if verdicts:
        images = [img for img in images if img.get("verdict") in verdicts]

    # Build ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            img_path = Path(img.get("image_path", ""))
            if img_path.exists():
                zf.write(img_path, img_path.name)

        if include_manifest:
            manifest = {
                "run_id": run_id,
                "intent": run.get("intent", ""),
                "filter": filter,
                "image_count": len(images),
                "images": [
                    {
                        "image_id": img.get("image_id"),
                        "filename": Path(img.get("image_path", "")).name,
                        "verdict": img.get("verdict"),
                        "scores": img.get("scores", {}),
                        "flags": img.get("flags", []),
                    }
                    for img in images
                ],
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        if include_report:
            report_json = run.get("report_json")
            if report_json:
                report_str = report_json if isinstance(report_json, str) else json.dumps(report_json, indent=2)
                zf.writestr("report.json", report_str)

    buf.seek(0)
    filename = f"validation-{run_id}-{filter}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
```

- [ ] **Step 3: Register router, run tests, commit**

```bash
git add api/routes/export.py tests/test_api/test_export.py api/server.py
git commit -m "feat: add export endpoint — download curated dataset as ZIP"
```

---

### Task 5: WebSocket — live event streaming

**Files:**
- Create: `api/routes/ws.py`
- Modify: `api/server.py` (register router)
- Test: `tests/test_api/test_ws.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api/test_ws.py
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.server import app
from api.routes.runs import _active_runs
from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import ModuleStarted, ModuleCompleted
import pytest


@pytest.fixture
def client():
    app.state.config = MagicMock()
    app.state.store = MagicMock()
    return TestClient(app)


def test_websocket_receives_events(client):
    # Create a bus and register it as active
    bus = EventBus()
    _active_runs["test-run"] = bus

    try:
        with client.websocket_connect("/api/runs/test-run/stream") as ws:
            # Publish an event on the bus
            bus.publish(ModuleStarted(module="executor", details="testing"))

            # Should receive it
            data = ws.receive_json()
            assert data["type"] == "ModuleStarted"
            assert data["module"] == "executor"
    finally:
        _active_runs.pop("test-run", None)


def test_websocket_run_not_active(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/api/runs/nonexistent/stream") as ws:
            pass
```

- [ ] **Step 2: Implement WebSocket router**

```python
# api/routes/ws.py
import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from api.routes.runs import get_active_bus

router = APIRouter(tags=["websocket"])


@router.websocket("/api/runs/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str):
    bus = get_active_bus(run_id)
    if not bus:
        await websocket.close(code=4004, reason="Run not active")
        return

    await websocket.accept()

    # Queue for bridging sync EventBus -> async WebSocket
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_event(event):
        data = json.loads(event.model_dump_json())
        data["type"] = type(event).__name__
        loop.call_soon_threadsafe(queue.put_nowait, data)

    bus.subscribe_all(on_event)

    try:
        while True:
            # Wait for events with timeout (detect disconnection)
            try:
                data = await asyncio.wait_for(queue.get(), timeout=60.0)
                await websocket.send_json(data)

                # If run completed, send final message and close
                if data.get("type") == "ModuleCompleted" and data.get("module") == "reporter":
                    await websocket.send_json({"type": "RunCompleted", "run_id": run_id})
                    break
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        # Clean up subscriber (best effort)
        pass
```

- [ ] **Step 3: Register router, run tests, commit**

```bash
git add api/routes/ws.py tests/test_api/test_ws.py api/server.py
git commit -m "feat: add WebSocket endpoint for live event streaming"
```

---

### Task 6: Run script for development

**Files:**
- Create: `api/run_dev.py`

- [ ] **Step 1: Create dev runner**

```python
# api/run_dev.py
"""Run the FastAPI development server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 2: Test it starts**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -c "import uvicorn; print('uvicorn OK')"`

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/sissi/Desktop/validation-pipeline && python3 -m pytest -v -m 'not integration'`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add api/run_dev.py
git commit -m "feat: add FastAPI dev server runner"
```

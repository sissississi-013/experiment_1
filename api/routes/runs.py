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

class CreateRunResponse(BaseModel):
    run_id: str
    status: str = "running"

_active_runs: dict[str, EventBus] = {}

def get_active_bus(run_id: str) -> EventBus | None:
    return _active_runs.get(run_id)

def run_pipeline_background(run_id: str, request: CreateRunRequest, config: PipelineConfig, store):
    bus = EventBus()
    if store:
        persistence = PersistenceSubscriber(store, run_id)
        bus.subscribe_all(persistence)
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
            config_json=config.model_dump() if hasattr(config, 'model_dump') else {},
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

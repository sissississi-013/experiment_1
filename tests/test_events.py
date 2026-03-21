import json
from datetime import datetime
from validation_pipeline.events import (
    PipelineEvent, ModuleStarted, ModuleCompleted,
    ImageProgress, ToolProgress, ImageVerdict,
    SpecGenerated, PlanGenerated, DatasetResolved,
    PipelineErrorEvent,
)

def test_pipeline_event_has_timestamp_and_module():
    event = PipelineEvent(module="test")
    assert event.module == "test"
    assert isinstance(event.timestamp, datetime)

def test_module_started_event():
    event = ModuleStarted(module="executor", details="processing 10 images")
    assert event.module == "executor"
    assert event.details == "processing 10 images"

def test_image_progress_event():
    event = ImageProgress(module="executor", current=3, total=10, image_path="/img/003.jpg")
    assert event.current == 3
    assert event.total == 10

def test_image_verdict_event():
    event = ImageVerdict(
        module="executor", image_id="003", image_path="/img/003.jpg",
        verdict="usable", scores={"blur": 0.95, "exposure": 0.6},
    )
    assert event.verdict == "usable"
    assert event.scores["blur"] == 0.95

def test_events_are_json_serializable():
    event = ModuleCompleted(module="spec_generator", duration_seconds=2.1, summary="done")
    data = json.loads(event.model_dump_json())
    assert data["module"] == "spec_generator"
    assert data["duration_seconds"] == 2.1

def test_pipeline_error_event():
    event = PipelineErrorEvent(
        module="executor", error_type="ToolError",
        message="API timeout", context={"tool": "nvidia"},
    )
    assert event.error_type == "ToolError"

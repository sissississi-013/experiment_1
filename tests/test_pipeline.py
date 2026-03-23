import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch
from validation_pipeline.config import PipelineConfig
from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.schemas.spec import FormalSpec, QualityCriterion, QuantityTarget, OutputFormat
from validation_pipeline.schemas.plan import ValidationPlan, PlanStep, SamplingStrategy, CostEstimate
from validation_pipeline.pipeline import ValidationPipeline
from validation_pipeline.schemas.dataset import DatasetPlan
from validation_pipeline.errors import DatasetError, SpecValidationError
from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import ModuleStarted, ModuleCompleted, SpecGenerated, PlanGenerated


def test_full_pipeline(tmp_path):
    img_dir = tmp_path / "dataset"
    img_dir.mkdir()
    for i in range(20):
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_dir / f"img_{i:03d}.jpg"))

    config_dir = tmp_path / "tool_configs"
    config_dir.mkdir()
    (config_dir / "laplacian_blur.yaml").write_text(
        "name: laplacian_blur\ntask_type: image_quality\ntier: 1\nsource: local\n"
        'wrapper_class: "opencv_wrapper.LaplacianBlurTool"\ndefault_config: {}\ncost_estimate_ms: 1\n'
    )

    mock_spec = FormalSpec(
        restated_request="Find sharp images",
        assumptions=["Test dataset"],
        content_criteria=[],
        quality_criteria=[QualityCriterion(dimension="blur", description="not blurry")],
        quantity_targets=QuantityTarget(),
        output_format=OutputFormat(),
        success_criteria="Sharp images",
        user_confirmed=True,
    )
    mock_plan = ValidationPlan(
        plan_id="plan_001", spec_summary="Find sharp images",
        sampling_strategy=SamplingStrategy(),
        steps=[PlanStep(
            step_id=1, dimension="blur", tool_name="laplacian_blur",
            strictness=0.5,
            hypothesis="Laplacian catches blur", tier=1,
        )],
        combination_logic="ALL_PASS",
        estimated_cost=CostEstimate(),
        user_approved=True,
    )

    config = PipelineConfig(tool_configs_dir=str(config_dir))
    pipeline = ValidationPipeline(config)

    with patch("validation_pipeline.modules.spec_generator._call_llm", return_value=mock_spec), \
         patch("validation_pipeline.modules.planner._call_llm", return_value=mock_plan):
        report = pipeline.run(
            UserInput(dataset_path=str(img_dir), intent="find sharp images"),
            auto_approve=True,
        )

    assert report.dataset_stats.total_images == 20
    assert report.curation_score.overall_score >= 0.0
    assert report.dataset_stats.usable + report.dataset_stats.recoverable + report.dataset_stats.unusable + report.dataset_stats.error_count == 20


def test_pipeline_with_dataset_description(tmp_path):
    """Pipeline resolves dataset_description when dataset_path is None."""
    img_dir = tmp_path / "resolved_dataset"
    img_dir.mkdir()
    for i in range(10):
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_dir / f"img_{i:03d}.jpg"))

    config_dir = tmp_path / "tool_configs"
    config_dir.mkdir()
    (config_dir / "laplacian_blur.yaml").write_text(
        "name: laplacian_blur\ntask_type: image_quality\ntier: 1\nsource: local\n"
        'wrapper_class: "opencv_wrapper.LaplacianBlurTool"\ndefault_config: {}\ncost_estimate_ms: 1\n'
    )

    mock_spec = FormalSpec(
        restated_request="Find sharp images",
        assumptions=["Test dataset"],
        content_criteria=[],
        quality_criteria=[QualityCriterion(dimension="blur", description="not blurry")],
        quantity_targets=QuantityTarget(),
        output_format=OutputFormat(),
        success_criteria="Sharp images",
        user_confirmed=True,
    )
    mock_plan = ValidationPlan(
        plan_id="plan_001", spec_summary="Find sharp images",
        sampling_strategy=SamplingStrategy(),
        steps=[PlanStep(
            step_id=1, dimension="blur", tool_name="laplacian_blur",
            strictness=0.5,
            hypothesis="Laplacian catches blur", tier=1,
        )],
        combination_logic="ALL_PASS",
        estimated_cost=CostEstimate(),
        user_approved=True,
    )

    mock_dataset_plan = DatasetPlan(
        source="coco", url="http://images.cocodataset.org",
        subset="val2017", category_filter=None,
        max_images=10, download_path=str(img_dir),
    )

    config = PipelineConfig(tool_configs_dir=str(config_dir))
    pipeline = ValidationPipeline(config)

    with patch("validation_pipeline.modules.spec_generator._call_llm", return_value=mock_spec), \
         patch("validation_pipeline.modules.planner._call_llm", return_value=mock_plan), \
         patch("validation_pipeline.modules.dataset_resolver._call_llm", return_value=mock_dataset_plan), \
         patch("validation_pipeline.modules.dataset_resolver.download_dataset", return_value=str(img_dir)):
        report = pipeline.run(
            UserInput(intent="find sharp images", dataset_description="10 images from COCO"),
            auto_approve=True,
        )

    assert report.dataset_stats.total_images == 10


def test_pipeline_raises_dataset_error_no_path_or_description():
    config = PipelineConfig(tool_configs_dir="/nonexistent")
    pipeline = ValidationPipeline(config)
    with pytest.raises(DatasetError):
        pipeline.run(UserInput(intent="test"), auto_approve=True)


def test_pipeline_raises_spec_validation_error_unconfirmed(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = PipelineConfig(tool_configs_dir=str(config_dir))
    pipeline = ValidationPipeline(config)
    mock_spec = FormalSpec(
        restated_request="test", assumptions=[], content_criteria=[],
        quality_criteria=[], quantity_targets=QuantityTarget(),
        output_format=OutputFormat(), success_criteria="test",
        user_confirmed=False,
    )
    with patch("validation_pipeline.modules.spec_generator._call_llm", return_value=mock_spec):
        with pytest.raises(SpecValidationError):
            pipeline.run(UserInput(dataset_path=str(img_dir), intent="test"), auto_approve=False)


def test_pipeline_publishes_lifecycle_events(tmp_path):
    img_dir = tmp_path / "dataset"
    img_dir.mkdir()
    for i in range(5):
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_dir / f"img_{i:03d}.jpg"))

    config_dir = tmp_path / "tool_configs"
    config_dir.mkdir()
    (config_dir / "laplacian_blur.yaml").write_text(
        "name: laplacian_blur\ntask_type: image_quality\ntier: 1\nsource: local\n"
        'wrapper_class: "opencv_wrapper.LaplacianBlurTool"\ndefault_config: {}\ncost_estimate_ms: 1\n'
    )

    mock_spec = FormalSpec(
        restated_request="Test", assumptions=[], content_criteria=[],
        quality_criteria=[QualityCriterion(dimension="blur", description="sharp")],
        quantity_targets=QuantityTarget(), output_format=OutputFormat(),
        success_criteria="test", user_confirmed=True,
    )
    mock_plan = ValidationPlan(
        plan_id="p1", spec_summary="test", sampling_strategy=SamplingStrategy(),
        steps=[PlanStep(step_id=1, dimension="blur", tool_name="laplacian_blur",
            strictness=0.5, hypothesis="test", tier=1)],
        combination_logic="ALL_PASS", estimated_cost=CostEstimate(), user_approved=True,
    )

    events_received = []
    bus = EventBus()
    bus.subscribe_all(lambda e: events_received.append(e))

    config = PipelineConfig(tool_configs_dir=str(config_dir))
    pipeline = ValidationPipeline(config, event_bus=bus)

    with patch("validation_pipeline.modules.spec_generator._call_llm", return_value=mock_spec), \
         patch("validation_pipeline.modules.planner._call_llm", return_value=mock_plan):
        pipeline.run(UserInput(dataset_path=str(img_dir), intent="test"), auto_approve=True)

    module_names = [e.module for e in events_received if isinstance(e, ModuleStarted)]
    assert "spec_generator" in module_names
    assert "executor" in module_names
    assert "recalibrator" in module_names
    assert "reporter" in module_names

    completed_names = [e.module for e in events_received if isinstance(e, ModuleCompleted)]
    assert "spec_generator" in completed_names
    assert "executor" in completed_names
    assert "recalibrator" in completed_names

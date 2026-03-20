import numpy as np
from PIL import Image
from unittest.mock import patch
from validation_pipeline.config import PipelineConfig
from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.schemas.spec import FormalSpec, QualityCriterion, QuantityTarget, OutputFormat
from validation_pipeline.schemas.plan import ValidationPlan, PlanStep, SamplingStrategy, CostEstimate
from validation_pipeline.pipeline import ValidationPipeline


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
            threshold=100.0, threshold_source="default",
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
    assert report.dataset_stats.usable + report.dataset_stats.recoverable + report.dataset_stats.unusable == 20

"""
End-to-end integration test on real COCO data.
Requires: API keys in .env, network access.
Run with: pytest tests/test_end_to_end.py -v -m integration
"""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()

from validation_pipeline.config import PipelineConfig
from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.pipeline import ValidationPipeline


pytestmark = pytest.mark.integration


@pytest.fixture
def pipeline():
    config = PipelineConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    return ValidationPipeline(config)


def test_end_to_end_coco_horses(pipeline, tmp_path):
    """Full pipeline: resolve COCO horses -> spec -> calibrate -> plan -> execute -> report."""
    user_input = UserInput(
        intent="find 10 sharp, well-exposed images of horses",
        dataset_description="10 horse images from COCO val2017",
    )

    report = pipeline.run(user_input, auto_approve=True)

    # Basic sanity checks
    assert report.dataset_stats.total_images > 0
    assert report.dataset_stats.total_images <= 10
    assert report.curation_score.overall_score >= 0.0
    assert report.dataset_stats.usable + report.dataset_stats.recoverable + report.dataset_stats.unusable == report.dataset_stats.total_images
    assert report.audit_trail is not None

    # Verify images were actually downloaded
    assert report.per_image_results is not None
    assert len(report.per_image_results) > 0

    # Verify at least some tool results exist per image
    for img_report in report.per_image_results:
        assert img_report.verdict in ("usable", "recoverable", "unusable")

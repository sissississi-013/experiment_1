import numpy as np
from PIL import Image
from validation_pipeline.schemas.spec import FormalSpec, QualityCriterion, QuantityTarget, OutputFormat
from validation_pipeline.schemas.calibration import CalibrationResult
from validation_pipeline.modules.calibrator import calibrate
from validation_pipeline.tools.wrappers.opencv_wrapper import LaplacianBlurTool


def test_calibrator_produces_result(tmp_path):
    good_paths = []
    for i in range(3):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        p = tmp_path / f"good_{i}.png"
        Image.fromarray(arr).save(str(p))
        good_paths.append(str(p))

    bad_paths = []
    for i in range(3):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        noise = np.random.randint(-2, 2, (100, 100, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        p = tmp_path / f"bad_{i}.png"
        Image.fromarray(arr).save(str(p))
        bad_paths.append(str(p))

    spec = FormalSpec(
        restated_request="Find sharp images",
        assumptions=[],
        content_criteria=[],
        quality_criteria=[QualityCriterion(dimension="blur", description="not blurry")],
        quantity_targets=QuantityTarget(),
        output_format=OutputFormat(),
        success_criteria="Sharp images",
    )

    tools = {"blur": LaplacianBlurTool(config={})}
    result = calibrate(spec, good_paths, bad_paths, tools)
    assert isinstance(result, CalibrationResult)
    assert "blur" in result.tool_calibrations
    cal = result.tool_calibrations["blur"]
    assert np.mean(cal.raw_good_scores) > np.mean(cal.raw_bad_scores)


from validation_pipeline.schemas.spec import FormalSpec, QualityCriterion, QuantityTarget, OutputFormat
from validation_pipeline.modules.calibrator import calibrate

def test_calibrator_no_exemplars_returns_empty_result():
    """With no exemplars, calibrator should return clean empty result without numpy warnings."""
    spec = FormalSpec(
        restated_request="test", assumptions=[],
        content_criteria=[],
        quality_criteria=[QualityCriterion(dimension="blur", description="sharp")],
        quantity_targets=QuantityTarget(),
        output_format=OutputFormat(),
        success_criteria="test",
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = calibrate(spec, [], [], {})
    assert result.tool_calibrations == {}
    assert len(result.threshold_report) == 1
    assert "No exemplars" in result.threshold_report[0].explanation

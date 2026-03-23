"""Integration tests: verify recalibrator produces reasonable verdicts
on synthetic datasets with known quality distributions."""
import numpy as np
from PIL import Image
from validation_pipeline.schemas.program import CompiledProgram, ProgramLine, BatchStrategy
from validation_pipeline.modules.executor import execute_program
from validation_pipeline.modules.recalibrator import recalibrate
from validation_pipeline.tools.wrappers.opencv_wrapper import LaplacianBlurTool


def test_recalibration_separates_blurry_from_sharp(tmp_path):
    """Sharp images should be usable, blurry should not."""
    for i in range(10):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(tmp_path / f"sharp_{i:03d}.jpg"))
    for i in range(10):
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        arr[:, :, 0] = np.linspace(120, 136, 100).astype(np.uint8)
        Image.fromarray(arr).save(str(tmp_path / f"blurry_{i:03d}.jpg"))

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[ProgramLine(
            line_number=1, variable_name="blur_score",
            tool_call="laplacian_blur(image)", output_type="float", tier=1,
        )],
        batch_strategy=BatchStrategy(early_exit=False),
        tool_imports=["laplacian_blur"],
    )
    tools = {"laplacian_blur": LaplacianBlurTool(config={})}
    execution = execute_program(program, str(tmp_path), tools)

    result = recalibrate(execution, strictness_hints={"blur": 0.5})

    sharp_verdicts = [
        result.image_verdicts[f"sharp_{i:03d}"].verdict for i in range(10)
    ]
    blurry_verdicts = [
        result.image_verdicts[f"blurry_{i:03d}"].verdict for i in range(10)
    ]
    assert sum(1 for v in sharp_verdicts if v == "usable") >= 7
    assert sum(1 for v in blurry_verdicts if v in ("recoverable", "unusable")) >= 7


def test_recalibration_reports_method_and_confidence(tmp_path):
    """Verify recalibration result includes method and confidence info."""
    for i in range(20):
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(tmp_path / f"img_{i:03d}.jpg"))

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[ProgramLine(
            line_number=1, variable_name="blur_score",
            tool_call="laplacian_blur(image)", output_type="float", tier=1,
        )],
        batch_strategy=BatchStrategy(early_exit=False),
        tool_imports=["laplacian_blur"],
    )
    tools = {"laplacian_blur": LaplacianBlurTool(config={})}
    execution = execute_program(program, str(tmp_path), tools)

    result = recalibrate(execution)

    assert "blur" in result.dimension_calibrations
    dc = result.dimension_calibrations["blur"]
    assert dc.method in ("gmm", "percentile")
    assert dc.confidence >= 0.0
    assert len(dc.thresholds) == 2
    assert dc.explanation != ""
    assert result.method_summary != ""

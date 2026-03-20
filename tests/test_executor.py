import numpy as np
from PIL import Image
from validation_pipeline.schemas.program import CompiledProgram, ProgramLine, BatchStrategy
from validation_pipeline.schemas.execution import ExecutionResult
from validation_pipeline.modules.executor import execute_program
from validation_pipeline.tools.wrappers.opencv_wrapper import LaplacianBlurTool


def test_executor_processes_all_images(tmp_path):
    for i in range(10):
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(tmp_path / f"img_{i:03d}.jpg"))

    program = CompiledProgram(
        program_id="prog_001",
        source_plan_id="plan_001",
        per_image_lines=[
            ProgramLine(line_number=1, variable_name="blur_score",
                        tool_call="laplacian_blur(image)",
                        output_type="float", threshold_check="blur_score >= 100.0", tier=1),
        ],
        batch_strategy=BatchStrategy(early_exit=True, error_policy="skip_and_log"),
        tool_imports=["laplacian_blur"],
    )

    tools = {"laplacian_blur": LaplacianBlurTool(config={})}
    result = execute_program(program, str(tmp_path), tools)

    assert isinstance(result, ExecutionResult)
    assert result.processed == 10
    assert len(result.results) == 10
    for img_result in result.results:
        assert img_result.verdict in ("usable", "recoverable", "unusable")
        assert img_result.verdict_reason != ""

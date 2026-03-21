import numpy as np
from PIL import Image
from validation_pipeline.schemas.program import CompiledProgram, ProgramLine, BatchStrategy
from validation_pipeline.schemas.execution import ExecutionResult, ToolResult
from validation_pipeline.modules.executor import execute_program
from validation_pipeline.tools.wrappers.opencv_wrapper import LaplacianBlurTool
from validation_pipeline.tools.base import BaseTool


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


def test_executor_passes_tool_params(tmp_path):
    """Verify that tool_params are forwarded to tool.execute()."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img_dir / "test.jpg"))

    received_kwargs = {}

    class MockContentTool(BaseTool):
        name = "mock_content"
        task_type = "content_detection"
        tier = 2

        def execute(self, image, **kwargs):
            received_kwargs.update(kwargs)
            return 0.9

        def normalize(self, raw_output, calibration=None):
            return ToolResult(
                tool_name="mock_content", dimension="content",
                score=raw_output, passed=True, threshold=0.5,
                raw_output=raw_output,
            )

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[ProgramLine(
            line_number=1, variable_name="content_score",
            tool_call="mock_content(image)", output_type="float",
            threshold_check="content_score >= 0.5", tier=2,
            tool_params={"target_label": "horse"},
        )],
        batch_strategy=BatchStrategy(),
        tool_imports=["mock_content"],
    )
    tools = {"mock_content": MockContentTool({})}
    result = execute_program(program, str(img_dir), tools)
    assert received_kwargs == {"target_label": "horse"}
    assert result.results[0].verdict == "usable"


def test_executor_uses_normalized_score_for_threshold(tmp_path):
    """Verify executor uses normalized score, not raw_output, for threshold."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img_dir / "test.jpg"))

    class DictReturningTool(BaseTool):
        name = "dict_tool"
        task_type = "content_detection"
        tier = 2

        def execute(self, image, **kwargs):
            return {"best_confidence": 0.9, "detections": []}

        def normalize(self, raw_output, calibration=None):
            return ToolResult(
                tool_name="dict_tool", dimension="content",
                score=raw_output["best_confidence"], passed=True,
                threshold=0.5, raw_output=raw_output,
            )

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[ProgramLine(
            line_number=1, variable_name="content_score",
            tool_call="dict_tool(image)", output_type="float",
            threshold_check="content_score >= 0.5", tier=2,
            tool_params={"target_label": "horse"},
        )],
        batch_strategy=BatchStrategy(),
        tool_imports=["dict_tool"],
    )
    tools = {"dict_tool": DictReturningTool({})}
    result = execute_program(program, str(img_dir), tools)
    assert result.results[0].verdict == "usable"


from validation_pipeline.errors import ToolError


def test_executor_collects_tool_errors(tmp_path):
    """When a tool raises ToolError, executor records it instead of crashing."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img_dir / "test.jpg"))

    class FailingTool(BaseTool):
        name = "failing_tool"
        task_type = "content_detection"
        tier = 2
        def execute(self, image, **kwargs):
            raise ToolError("API timeout", module="failing_tool", context={"http_status": 500})
        def normalize(self, raw_output, calibration=None):
            pass

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[ProgramLine(
            line_number=1, variable_name="content_score",
            tool_call="failing_tool(image)", output_type="float",
            threshold_check="content_score >= 0.5", tier=2,
        )],
        batch_strategy=BatchStrategy(),
        tool_imports=["failing_tool"],
    )
    tools = {"failing_tool": FailingTool({})}
    result = execute_program(program, str(img_dir), tools)

    assert result.results[0].verdict == "error"
    assert len(result.results[0].errors) > 0
    assert "API timeout" in result.results[0].errors[0]
    assert result.summary.error_count == 1


def test_executor_partial_tool_failure(tmp_path):
    """When some tools succeed and some fail, verdict uses available results."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img_dir / "test.jpg"))

    class GoodTool(BaseTool):
        name = "good_tool"
        task_type = "image_quality"
        tier = 1
        def execute(self, image, **kwargs):
            return 0.9
        def normalize(self, raw_output, calibration=None):
            return ToolResult(
                tool_name="good_tool", dimension="blur",
                score=0.9, passed=True, threshold=0.5, raw_output=0.9,
            )

    class BadTool(BaseTool):
        name = "bad_tool"
        task_type = "content_detection"
        tier = 2
        def execute(self, image, **kwargs):
            raise ToolError("timeout", module="bad_tool")
        def normalize(self, raw_output, calibration=None):
            pass

    program = CompiledProgram(
        program_id="p1", source_plan_id="plan1",
        per_image_lines=[
            ProgramLine(line_number=1, variable_name="blur_score", tool_call="good_tool(image)", output_type="float", threshold_check="blur_score >= 0.5", tier=1),
            ProgramLine(line_number=2, variable_name="content_score", tool_call="bad_tool(image)", output_type="float", threshold_check="content_score >= 0.5", tier=2),
        ],
        batch_strategy=BatchStrategy(early_exit=False),
        tool_imports=["good_tool", "bad_tool"],
    )
    tools = {"good_tool": GoodTool({}), "bad_tool": BadTool({})}
    result = execute_program(program, str(img_dir), tools)

    img = result.results[0]
    assert len(img.errors) > 0
    assert len(img.tool_results) == 1
    assert img.verdict != "error"

from pydantic import BaseModel


class ProgramLine(BaseModel):
    line_number: int
    variable_name: str
    tool_call: str
    output_type: str
    threshold_check: str | None = None
    tier: int = 1


class BatchStrategy(BaseModel):
    parallelism: int = 4
    tier1_batch_size: int = 100
    tier2_batch_size: int = 16
    tier3_batch_size: int = 1
    early_exit: bool = True
    error_policy: str = "skip_and_log"


class CompiledProgram(BaseModel):
    program_id: str
    source_plan_id: str
    per_image_lines: list[ProgramLine]
    batch_strategy: BatchStrategy
    tool_imports: list[str]

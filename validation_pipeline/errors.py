class PipelineError(Exception):
    """Base for all pipeline errors."""
    def __init__(self, message: str, module: str, context: dict | None = None):
        self.module = module
        self.context = context or {}
        super().__init__(message)


class DatasetError(PipelineError):
    """Download failures, missing paths, bad formats, category not found."""
    pass


class LLMError(PipelineError):
    """OpenAI/instructor failures, invalid structured responses, retry exhaustion."""
    pass


class ToolError(PipelineError):
    """Tool execution failures — API timeout, bad image, rate limiting."""
    pass


class CalibrationError(PipelineError):
    """Not enough exemplars, degenerate data, failed Platt fitting."""
    pass


class SpecValidationError(PipelineError):
    """Spec/plan validation failures — missing fields, invalid values."""
    pass

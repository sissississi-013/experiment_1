from validation_pipeline.errors import (
    PipelineError, DatasetError, LLMError, ToolError,
    CalibrationError, SpecValidationError,
)


def test_pipeline_error_has_module_and_context():
    err = PipelineError("something broke", module="executor", context={"image": "foo.jpg"})
    assert str(err) == "something broke"
    assert err.module == "executor"
    assert err.context == {"image": "foo.jpg"}


def test_pipeline_error_default_context():
    err = PipelineError("fail", module="test")
    assert err.context == {}


def test_tool_error_is_pipeline_error():
    err = ToolError("API timeout", module="nvidia_grounding_dino", context={"http_status": 500})
    assert isinstance(err, PipelineError)
    assert err.module == "nvidia_grounding_dino"


def test_llm_error_is_pipeline_error():
    err = LLMError("Invalid response", module="spec_generator")
    assert isinstance(err, PipelineError)


def test_dataset_error_is_pipeline_error():
    err = DatasetError("Download failed", module="dataset_resolver", context={"url": "http://x.com"})
    assert isinstance(err, PipelineError)


def test_calibration_error_is_pipeline_error():
    err = CalibrationError("No exemplars", module="calibrator")
    assert isinstance(err, PipelineError)


def test_spec_validation_error_is_pipeline_error():
    err = SpecValidationError("Plan not approved", module="compiler")
    assert isinstance(err, PipelineError)

import time
from unittest.mock import patch
from validation_pipeline.retry import retry_with_policy
from validation_pipeline.config import RetryPolicy
from validation_pipeline.errors import ToolError


def test_retry_succeeds_first_try():
    result = retry_with_policy(
        fn=lambda: "ok",
        policy=RetryPolicy(),
        error_cls=ToolError,
        module="test",
    )
    assert result == "ok"


def test_retry_succeeds_after_transient_failure():
    call_count = {"n": 0}
    def flaky():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("transient")
        return "recovered"

    with patch("time.sleep"):
        result = retry_with_policy(
            fn=flaky,
            policy=RetryPolicy(max_retries=3),
            error_cls=ToolError,
            module="test",
        )
    assert result == "recovered"
    assert call_count["n"] == 2


def test_retry_raises_on_exhaustion():
    def always_fails():
        raise TimeoutError("timeout")

    import pytest
    with patch("time.sleep"):
        with pytest.raises(ToolError) as exc_info:
            retry_with_policy(
                fn=always_fails,
                policy=RetryPolicy(max_retries=2),
                error_cls=ToolError,
                module="nvidia_grounding_dino",
                context={"target_label": "horse"},
            )
    assert exc_info.value.module == "nvidia_grounding_dino"
    assert "retry_count" in exc_info.value.context
    assert exc_info.value.context["retry_count"] == 2


def test_retry_raises_immediately_on_permanent_error():
    from requests.exceptions import HTTPError
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.status_code = 401

    call_count = {"n": 0}
    def auth_fail():
        call_count["n"] += 1
        err = HTTPError(response=resp)
        raise err

    import pytest
    with patch("time.sleep"):
        with pytest.raises(ToolError):
            retry_with_policy(
                fn=auth_fail,
                policy=RetryPolicy(max_retries=3),
                error_cls=ToolError,
                module="test",
            )
    assert call_count["n"] == 1


def test_retry_exponential_backoff():
    sleep_calls = []
    def always_fails():
        raise ConnectionError("fail")

    import pytest
    with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        with pytest.raises(ToolError):
            retry_with_policy(
                fn=always_fails,
                policy=RetryPolicy(max_retries=3, base_delay=1.0, backoff_factor=2.0),
                error_cls=ToolError,
                module="test",
            )
    assert sleep_calls == [1.0, 2.0, 4.0]

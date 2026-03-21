import time
from typing import TypeVar, Callable
from requests.exceptions import HTTPError
from validation_pipeline.errors import PipelineError

T = TypeVar("T")

PERMANENT_STATUS_CODES = {400, 401, 403, 404, 422}


def _is_permanent_http_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError) and exc.response is not None:
        return exc.response.status_code in PERMANENT_STATUS_CODES
    return False


def retry_with_policy(
    fn: Callable[[], T],
    policy,
    error_cls: type[PipelineError],
    module: str,
    context: dict | None = None,
) -> T:
    context = context or {}

    for attempt in range(policy.max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if _is_permanent_http_error(e):
                raise error_cls(
                    f"Permanent error: {e}",
                    module=module,
                    context={**context, "retry_count": attempt, "error_type": type(e).__name__},
                ) from e

            if attempt == policy.max_retries:
                raise error_cls(
                    f"Retry exhausted after {policy.max_retries + 1} attempts: {e}",
                    module=module,
                    context={**context, "retry_count": policy.max_retries, "error_type": type(e).__name__},
                ) from e

            delay = min(
                policy.base_delay * (policy.backoff_factor ** attempt),
                policy.max_delay,
            )
            time.sleep(delay)

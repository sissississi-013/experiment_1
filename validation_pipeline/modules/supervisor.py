from validation_pipeline.schemas.execution import ExecutionResult
from validation_pipeline.schemas.calibration import CalibrationResult
from validation_pipeline.schemas.plan import ValidationPlan
from validation_pipeline.schemas.supervision import SupervisionReport, SupervisionCheck, Anomaly


def supervise(
    result: ExecutionResult,
    calibration: CalibrationResult,
    plan: ValidationPlan,
) -> SupervisionReport:
    checks = []
    anomalies = []

    for dim, rate in result.summary.flag_rates.items():
        reasonable = rate < 0.6
        checks.append(SupervisionCheck(
            check_name=f"flag_rate_{dim}",
            passed=reasonable,
            details=f"{dim} flag rate: {rate:.1%}",
        ))
        if not reasonable:
            anomalies.append(Anomaly(
                severity="warning" if rate < 0.85 else "blocker",
                description=f"{dim} flagged {rate:.0%} of images",
                likely_cause="Threshold may be too aggressive or dataset quality is genuinely low",
                suggested_action=f"Review {dim} threshold or inspect sample of flagged images",
            ))

    for tool, rate in result.summary.tool_error_rate.items():
        ok = rate < 0.05
        checks.append(SupervisionCheck(
            check_name=f"error_rate_{tool}",
            passed=ok,
            details=f"{tool} error rate: {rate:.1%}",
        ))
        if not ok:
            anomalies.append(Anomaly(
                severity="warning",
                description=f"{tool} errored on {rate:.0%} of images",
                likely_cause="Images may be corrupt or incompatible with tool",
                suggested_action=f"Check image formats and sizes for {tool} compatibility",
            ))

    # Check for high image error rate
    error_count = result.summary.error_count
    if result.total_images > 0:
        error_rate = error_count / result.total_images
        checks.append(SupervisionCheck(
            check_name="image_error_rate",
            passed=error_rate < 0.1,
            details=f"Image error rate: {error_rate:.1%} ({error_count}/{result.total_images})",
        ))
        if error_rate >= 0.1:
            anomalies.append(Anomaly(
                severity="blocker" if error_rate > 0.5 else "warning",
                description=f"{error_count} images failed with tool errors ({error_rate:.0%})",
                likely_cause="Tool API failures or incompatible image formats",
                suggested_action="Check API keys, network connectivity, and image formats",
            ))

    empty = result.summary.usable_count == 0 and result.processed > 0
    checks.append(SupervisionCheck(
        check_name="empty_result",
        passed=not empty,
        details=f"Usable: {result.summary.usable_count}/{result.processed}",
    ))
    if empty:
        anomalies.append(Anomaly(
            severity="blocker",
            description="Zero images passed all checks",
            likely_cause="Thresholds are too strict or tools are misconfigured",
            suggested_action="Revise thresholds or review the validation plan",
        ))

    has_blockers = any(a.severity == "blocker" for a in anomalies)
    has_warnings = any(a.severity == "warning" for a in anomalies)

    if has_blockers:
        status = "blocked"
        rec = "Execution blocked. Review anomalies and revise plan before proceeding."
    elif has_warnings:
        status = "warnings"
        rec = "Some anomalies detected. Review before delivering results."
    else:
        status = "passed"
        rec = "Results look clean. Ready for delivery."

    return SupervisionReport(
        status=status, checks=checks, anomalies=anomalies, recommendation=rec,
    )

from validation_pipeline.schemas.execution import ExecutionResult
from validation_pipeline.schemas.recalibration import RecalibrationResult
from validation_pipeline.schemas.calibration import CalibrationResult
from validation_pipeline.schemas.plan import ValidationPlan
from validation_pipeline.schemas.supervision import SupervisionReport, SupervisionCheck, Anomaly


def supervise(
    result: ExecutionResult,
    recalibration: RecalibrationResult,
    calibration: CalibrationResult,
    plan: ValidationPlan,
) -> SupervisionReport:
    checks = []
    anomalies = []

    # Compute flag rates from recalibration verdicts
    flag_rates: dict[str, float] = {}
    total = len(recalibration.image_verdicts) or 1
    for vr in recalibration.image_verdicts.values():
        for dim in vr.failed_dimensions:
            flag_rates[dim] = flag_rates.get(dim, 0) + 1
    flag_rates = {k: v / total for k, v in flag_rates.items()}

    for dim, rate in flag_rates.items():
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

    usable_count = sum(1 for v in recalibration.image_verdicts.values() if v.verdict == "usable")
    empty = usable_count == 0 and result.processed > 0
    checks.append(SupervisionCheck(
        check_name="empty_result",
        passed=not empty,
        details=f"Usable: {usable_count}/{result.processed}",
    ))
    if empty:
        anomalies.append(Anomaly(
            severity="blocker",
            description="Zero images passed all checks",
            likely_cause="Thresholds are too strict or tools are misconfigured",
            suggested_action="Revise thresholds or review the validation plan",
        ))

    # Check recalibration confidence
    for dim, dc in recalibration.dimension_calibrations.items():
        if dc.confidence < 0.6:
            anomalies.append(Anomaly(
                severity="warning",
                description=f"Calibration confidence is low for {dim} ({dc.confidence:.2f})",
                likely_cause="Score distribution lacks clear clusters",
                suggested_action="Consider providing exemplar images",
            ))

    if all(dc.method == "percentile" for dc in recalibration.dimension_calibrations.values()):
        anomalies.append(Anomaly(
            severity="warning",
            description="All dimensions used percentile-based thresholds",
            likely_cause="Score distributions were unimodal across all dimensions",
            suggested_action="Thresholds are relative to this batch, not absolute quality",
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

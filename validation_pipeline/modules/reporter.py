import uuid
from datetime import datetime
from validation_pipeline.schemas.execution import ExecutionResult
from validation_pipeline.schemas.recalibration import RecalibrationResult
from validation_pipeline.schemas.supervision import SupervisionReport
from validation_pipeline.schemas.spec import FormalSpec
from validation_pipeline.schemas.plan import ValidationPlan
from validation_pipeline.schemas.report import (
    FinalReport, DatasetStats, CurationScore, ImageReport, AuditTrail, OutputFiles,
)


def generate_report(
    result: ExecutionResult,
    recalibration: RecalibrationResult,
    supervision: SupervisionReport,
    spec: FormalSpec,
    plan: ValidationPlan,
) -> FinalReport:
    total = result.total_images or 1

    usable = sum(1 for v in recalibration.image_verdicts.values() if v.verdict == "usable")
    recoverable = sum(1 for v in recalibration.image_verdicts.values() if v.verdict == "recoverable")
    unusable = sum(1 for v in recalibration.image_verdicts.values() if v.verdict == "unusable")
    error_count = sum(1 for v in recalibration.image_verdicts.values() if v.verdict == "error")

    flag_breakdown: dict[str, int] = {}
    for vr in recalibration.image_verdicts.values():
        for dim in vr.failed_dimensions:
            flag_breakdown[dim] = flag_breakdown.get(dim, 0) + 1

    dim_scores = {}
    for dim in flag_breakdown:
        dim_scores[dim] = 1.0 - (flag_breakdown[dim] / total)

    overall = usable / total if total > 0 else 0.0

    dims_str = ", ".join(f"{d}: {s:.0%}" for d, s in dim_scores.items())
    explanation = f"{overall:.0%} of images meet all criteria."
    if dims_str:
        explanation += f" Per-dimension pass rates: {dims_str}."

    per_image = []
    for img in result.results:
        scores = {tr.dimension: tr.score for tr in img.tool_results}
        vr = recalibration.image_verdicts.get(img.image_id)
        flags = vr.failed_dimensions if vr else []
        verdict = vr.verdict if vr else img.verdict
        per_image.append(ImageReport(
            image_id=img.image_id,
            image_path=img.image_path,
            verdict=verdict,
            scores=scores,
            flags=flags,
            explanation=vr.explanation if vr else None,
        ))

    return FinalReport(
        report_id=str(uuid.uuid4())[:8],
        spec_summary=spec.restated_request,
        dataset_stats=DatasetStats(
            total_images=total,
            usable=usable,
            recoverable=recoverable,
            unusable=unusable,
            error_count=error_count,
            usable_percentage=usable / total,
            flag_breakdown=flag_breakdown,
        ),
        curation_score=CurationScore(
            overall_score=overall,
            dimension_scores=dim_scores,
            confidence=0.9 if supervision.status == "passed" else 0.5,
            explanation=explanation,
        ),
        per_image_results=per_image,
        audit_trail=AuditTrail(
            spec=spec.model_dump(),
            plan=plan.model_dump(),
            calibration={},
            recalibration=recalibration.model_dump(),
            supervision_report=supervision.model_dump(),
            timestamp=datetime.now().isoformat(),
            llm_model_used="gpt-4o",
            tool_versions={},
        ),
        output_files=OutputFiles(
            usable_manifest="output/usable.json",
            full_results_json="output/full_results.json",
        ),
    )

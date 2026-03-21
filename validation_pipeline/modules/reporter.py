import uuid
from datetime import datetime
from validation_pipeline.schemas.execution import ExecutionResult
from validation_pipeline.schemas.supervision import SupervisionReport
from validation_pipeline.schemas.spec import FormalSpec
from validation_pipeline.schemas.plan import ValidationPlan
from validation_pipeline.schemas.report import (
    FinalReport, DatasetStats, CurationScore, ImageReport, AuditTrail, OutputFiles,
)


def generate_report(
    result: ExecutionResult,
    supervision: SupervisionReport,
    spec: FormalSpec,
    plan: ValidationPlan,
) -> FinalReport:
    total = result.total_images or 1
    usable = result.summary.usable_count
    recoverable = result.summary.recoverable_count
    unusable = result.summary.unusable_count
    error_count = result.summary.error_count

    dim_scores = {}
    for dim, rate in result.summary.flag_rates.items():
        dim_scores[dim] = 1.0 - rate

    overall = usable / total if total > 0 else 0.0

    dims_str = ", ".join(f"{d}: {s:.0%}" for d, s in dim_scores.items())
    explanation = f"{overall:.0%} of images meet all criteria."
    if dims_str:
        explanation += f" Per-dimension pass rates: {dims_str}."

    per_image = []
    for img in result.results:
        scores = {tr.dimension: tr.score for tr in img.tool_results}
        flags = [tr.dimension for tr in img.tool_results if not tr.passed]
        per_image.append(ImageReport(
            image_id=img.image_id,
            image_path=img.image_path,
            verdict=img.verdict,
            scores=scores,
            flags=flags,
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
            flag_breakdown={d: int(r * total) for d, r in result.summary.flag_rates.items()},
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

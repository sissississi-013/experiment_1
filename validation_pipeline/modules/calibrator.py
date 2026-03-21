import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from validation_pipeline.schemas.spec import FormalSpec
from validation_pipeline.schemas.calibration import (
    CalibrationResult, ToolCalibration, ThresholdExplanation, EmbeddingRecord,
)
from validation_pipeline.tools.base import BaseTool


def calibrate(
    spec: FormalSpec,
    good_paths: list[str],
    bad_paths: list[str],
    tools: dict[str, BaseTool],
) -> CalibrationResult:
    tool_calibrations = {}
    threshold_report = []

    # No exemplars: skip calibration entirely
    if not good_paths and not bad_paths:
        return CalibrationResult(
            tool_calibrations={},
            exemplar_embeddings=[],
            threshold_report=[
                ThresholdExplanation(
                    dimension=qc.dimension,
                    threshold=0.5,
                    explanation="No exemplars provided. Using default thresholds.",
                )
                for qc in spec.quality_criteria
            ],
        )

    for criterion in spec.quality_criteria:
        dim = criterion.dimension
        if dim not in tools:
            continue
        tool = tools[dim]

        good_scores = []
        for p in good_paths:
            try:
                good_scores.append(tool.execute(Image.open(p).convert("RGB")))
            except Exception:
                pass

        bad_scores = []
        for p in bad_paths:
            try:
                bad_scores.append(tool.execute(Image.open(p).convert("RGB")))
            except Exception:
                pass

        if not good_scores and not bad_scores:
            from validation_pipeline.errors import CalibrationError
            raise CalibrationError(
                f"All exemplar images failed for dimension '{dim}'",
                module="calibrator",
                context={"dimension": dim, "good_count": len(good_paths), "bad_count": len(bad_paths)},
            )

        cal = _fit_platt(dim, tool.name, good_scores, bad_scores)
        tool_calibrations[dim] = cal

        threshold_report.append(ThresholdExplanation(
            dimension=dim,
            threshold=cal.calibrated_threshold,
            explanation=(
                f"Good examples averaged {np.mean(good_scores):.2f}, "
                f"bad averaged {np.mean(bad_scores):.2f}. "
                f"Threshold set at {cal.calibrated_threshold:.2f}."
            ),
        ))

    return CalibrationResult(
        tool_calibrations=tool_calibrations,
        exemplar_embeddings=[],
        threshold_report=threshold_report,
    )


def _fit_platt(
    dimension: str,
    tool_name: str,
    good_scores: list[float],
    bad_scores: list[float],
) -> ToolCalibration:
    all_scores = good_scores + bad_scores
    labels = [1] * len(good_scores) + [0] * len(bad_scores)

    X = np.array(all_scores).reshape(-1, 1)
    y = np.array(labels)

    if len(set(y)) < 2 or len(X) < 4:
        threshold = float(np.mean(all_scores))
        return ToolCalibration(
            tool_name=tool_name,
            raw_good_scores=good_scores,
            raw_bad_scores=bad_scores,
            calibrated_threshold=threshold,
            separability=0.0,
        )

    clf = LogisticRegression()
    clf.fit(X, y)

    platt_a = float(clf.coef_[0][0])
    platt_b = float(clf.intercept_[0])
    threshold = float(-platt_b / platt_a) if platt_a != 0 else float(np.mean(all_scores))

    good_mean = float(np.mean(good_scores))
    bad_mean = float(np.mean(bad_scores))
    combined_std = float(np.std(all_scores)) or 1.0
    separability = abs(good_mean - bad_mean) / combined_std

    return ToolCalibration(
        tool_name=tool_name,
        raw_good_scores=good_scores,
        raw_bad_scores=bad_scores,
        platt_a=platt_a,
        platt_b=platt_b,
        calibrated_threshold=threshold,
        separability=float(min(separability, 1.0)),
    )

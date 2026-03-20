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

    for criterion in spec.quality_criteria:
        dim = criterion.dimension
        if dim not in tools:
            continue
        tool = tools[dim]

        good_scores = [tool.execute(Image.open(p).convert("RGB")) for p in good_paths]
        bad_scores = [tool.execute(Image.open(p).convert("RGB")) for p in bad_paths]

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

import numpy as np
from sklearn.mixture import GaussianMixture
from validation_pipeline.schemas.recalibration import (
    DimensionCalibration, ImageVerdictRecord, RecalibrationResult,
)
from validation_pipeline.schemas.execution import ExecutionResult
from validation_pipeline.schemas.calibration import CalibrationResult

try:
    import diptest
    HAS_DIPTEST = True
except ImportError:
    HAS_DIPTEST = False


def _compute_gvf(scores: list[float], labels: list[int]) -> float:
    """Goodness of Variance Fit: 1.0 = perfect clustering, 0.0 = no clustering."""
    arr = np.array(scores)
    lab = np.array(labels)
    total_var = np.var(arr)
    if total_var == 0:
        return 0.0
    within_var = sum(
        np.var(arr[lab == k]) * np.sum(lab == k)
        for k in np.unique(lab)
    ) / len(arr)
    return 1.0 - (within_var / total_var)


def _gmm_crossing_points(gmm: GaussianMixture) -> list[float]:
    """Find crossing points between adjacent Gaussian components, sorted by mean."""
    means = gmm.means_.flatten()
    order = np.argsort(means)
    sorted_means = means[order]

    crossings = []
    for i in range(len(sorted_means) - 1):
        # Approximate crossing as weighted midpoint
        m1, m2 = sorted_means[i], sorted_means[i + 1]
        s1 = np.sqrt(gmm.covariances_.flatten()[order[i]])
        s2 = np.sqrt(gmm.covariances_.flatten()[order[i + 1]])
        # Weighted by inverse std — tighter cluster pulls crossing closer
        if s1 + s2 > 0:
            crossing = (m1 * s2 + m2 * s1) / (s1 + s2)
        else:
            crossing = (m1 + m2) / 2
        crossings.append(float(crossing))
    return crossings


def _analyze_dimension(
    dimension: str,
    scores: list[float],
    strictness: float = 0.5,
    exemplar_threshold: float | None = None,
    exemplar_separability: float | None = None,
) -> DimensionCalibration:
    """Analyze a single dimension's score distribution and determine thresholds."""
    n = len(scores)
    arr = np.array(scores)

    # Edge case: exemplars with good separability
    if exemplar_threshold is not None and exemplar_separability is not None:
        if exemplar_separability >= 0.3:
            return DimensionCalibration(
                dimension=dimension, method="exemplar",
                thresholds=[exemplar_threshold * 0.7, exemplar_threshold],
                confidence=min(exemplar_separability, 1.0),
                dip_test_p=-1.0, gmm_means=[],
                explanation=f"Using exemplar-derived threshold (separability {exemplar_separability:.2f})",
                strictness=strictness,
            )

    # Edge case: tiny batch
    if n < 5:
        return DimensionCalibration(
            dimension=dimension, method="percentile",
            thresholds=[0.3, 0.5],
            confidence=0.0, dip_test_p=-1.0, gmm_means=[],
            explanation=f"Insufficient data for calibration (n={n}). Using defaults.",
            strictness=strictness,
        )

    # Edge case: no variance
    if np.std(arr) < 1e-6:
        val = float(arr[0])
        verdict_hint = "usable" if val > 0.3 else "unusable"
        return DimensionCalibration(
            dimension=dimension, method="percentile",
            thresholds=[val - 0.01, val + 0.01],
            confidence=0.0, dip_test_p=-1.0, gmm_means=[],
            explanation=f"No score variation (all ~{val:.2f}). All images will be {verdict_hint}.",
            strictness=strictness,
        )

    # Percentile-based thresholds (used as fallback or for small/unimodal batches)
    unusable_pctl = 5 + 40 * strictness
    usable_pctl = 25 + 50 * strictness
    pctl_thresholds = [
        float(np.percentile(arr, unusable_pctl)),
        float(np.percentile(arr, usable_pctl)),
    ]

    # Small batch: skip dip test, use percentile
    if n < 10:
        return DimensionCalibration(
            dimension=dimension, method="percentile",
            thresholds=pctl_thresholds,
            confidence=0.3, dip_test_p=-1.0, gmm_means=[],
            explanation=f"Small batch (n={n}). Using percentile thresholds (strictness {strictness:.1f}).",
            strictness=strictness,
        )

    # Dip test for multimodality
    dip_p = 1.0
    if HAS_DIPTEST:
        _, dip_p = diptest.diptest(arr)
    else:
        dip_p = 1.0  # No diptest available, assume unimodal

    # Try GMM if multimodal
    if dip_p < 0.05 and n >= 10:
        try:
            # Compare BIC: 3 components vs 1
            gmm1 = GaussianMixture(n_components=1, random_state=42).fit(arr.reshape(-1, 1))
            gmm3 = GaussianMixture(n_components=3, random_state=42).fit(arr.reshape(-1, 1))

            if gmm3.bic(arr.reshape(-1, 1)) < gmm1.bic(arr.reshape(-1, 1)):
                crossings = _gmm_crossing_points(gmm3)
                labels = gmm3.predict(arr.reshape(-1, 1))
                gvf = _compute_gvf(scores, list(labels))
                means = sorted(gmm3.means_.flatten().tolist())

                if len(crossings) >= 2:
                    thresholds = [crossings[0], crossings[1]]
                elif len(crossings) == 1:
                    thresholds = [crossings[0] * 0.7, crossings[0]]
                else:
                    thresholds = pctl_thresholds

                means_str = ", ".join(f"{m:.2f}" for m in means)
                return DimensionCalibration(
                    dimension=dimension, method="gmm",
                    thresholds=thresholds, confidence=gvf,
                    dip_test_p=dip_p, gmm_means=means,
                    explanation=f"{dimension.capitalize()} scores formed 3 clusters centered at [{means_str}] (GMM, confidence {gvf:.2f}).",
                    strictness=strictness,
                )
        except Exception:
            pass  # Fall through to percentile

    # Unimodal or GMM failed: use percentile
    gvf = 0.0
    try:
        median = float(np.median(arr))
        labels = [0 if s < median else 1 for s in scores]
        gvf = _compute_gvf(scores, labels)
    except Exception:
        pass

    return DimensionCalibration(
        dimension=dimension, method="percentile",
        thresholds=pctl_thresholds, confidence=gvf,
        dip_test_p=dip_p, gmm_means=[],
        explanation=f"{dimension.capitalize()} scores are unimodal — using percentile thresholds (strictness {strictness:.1f}).",
        strictness=strictness,
    )


def _assign_verdicts(
    results: list,
    dim_calibrations: dict[str, DimensionCalibration],
) -> dict[str, ImageVerdictRecord]:
    """Assign verdicts to each image based on recalibrated thresholds.

    Two thresholds per dimension: [unusable_upper, recoverable_upper]
    - score < unusable_upper -> hard fail (unusable-level)
    - unusable_upper <= score < recoverable_upper -> soft fail (recoverable-level)
    - score >= recoverable_upper -> pass
    """
    verdicts = {}
    for img in results:
        scores = {tr.dimension: tr.score for tr in img.tool_results}
        hard_fails = []  # below unusable threshold
        soft_fails = []  # between unusable and usable threshold
        for dim, score in scores.items():
            if dim in dim_calibrations:
                cal = dim_calibrations[dim]
                unusable_t = cal.thresholds[0] if len(cal.thresholds) > 0 else 0.3
                usable_t = cal.thresholds[1] if len(cal.thresholds) > 1 else 0.5
                if score < unusable_t:
                    hard_fails.append(dim)
                elif score < usable_t:
                    soft_fails.append(dim)

        all_fails = hard_fails + soft_fails

        if not img.tool_results and img.errors:
            verdict = "error"
            explanation = "All tools failed: " + "; ".join(img.errors)
        elif len(all_fails) == 0:
            verdict = "usable"
            explanation = "All checks passed"
        elif len(hard_fails) >= 2 or (len(hard_fails) >= 1 and len(soft_fails) >= 1):
            verdict = "unusable"
            parts = []
            for dim in all_fails:
                score = scores.get(dim, 0)
                cal = dim_calibrations[dim]
                t = cal.thresholds[0] if dim in hard_fails else cal.thresholds[1]
                parts.append(f"{dim} ({score:.2f} < {t:.2f})")
            explanation = "Failed: " + "; ".join(parts)
        else:
            # 1 soft fail, or 1 hard fail with no other fails
            verdict = "recoverable"
            dim = all_fails[0]
            score = scores.get(dim, 0)
            cal = dim_calibrations[dim]
            t = cal.thresholds[0] if dim in hard_fails else cal.thresholds[1]
            explanation = f"Failed {dim} ({score:.2f} < {t:.2f})"

        verdicts[img.image_id] = ImageVerdictRecord(
            image_id=img.image_id,
            image_path=img.image_path,
            verdict=verdict,
            scores=scores,
            failed_dimensions=all_fails,
            explanation=explanation,
        )
    return verdicts


def recalibrate(
    execution: ExecutionResult,
    strictness_hints: dict[str, float] | None = None,
    calibration: CalibrationResult | None = None,
    event_bus=None,
) -> RecalibrationResult:
    """Main entry point: analyze batch distributions and assign verdicts."""
    try:
        from validation_pipeline.events import (
            RecalibrationStarted, ThresholdDetermined,
            RecalibrationCompleted, ImageVerdict,
        )
        has_events = True
    except ImportError:
        has_events = False

    strictness_hints = strictness_hints or {}

    # Collect scores per dimension across all images
    dim_scores: dict[str, list[float]] = {}
    for img in execution.results:
        for tr in img.tool_results:
            dim_scores.setdefault(tr.dimension, []).append(tr.score)

    dimensions = list(dim_scores.keys())
    if event_bus and has_events:
        event_bus.publish(RecalibrationStarted(module="recalibrator", dimensions=dimensions))

    # Analyze each dimension
    dim_calibrations: dict[str, DimensionCalibration] = {}
    for dim, scores in dim_scores.items():
        strictness = strictness_hints.get(dim, 0.5)

        # Check for exemplar calibration
        exemplar_threshold = None
        exemplar_sep = None
        if calibration and dim in calibration.tool_calibrations:
            tc = calibration.tool_calibrations[dim]
            exemplar_threshold = tc.calibrated_threshold
            exemplar_sep = tc.separability

        dc = _analyze_dimension(
            dim, scores, strictness,
            exemplar_threshold=exemplar_threshold,
            exemplar_separability=exemplar_sep,
        )
        dim_calibrations[dim] = dc

        if event_bus and has_events:
            event_bus.publish(ThresholdDetermined(
                module="recalibrator", dimension=dim,
                method=dc.method, thresholds=dc.thresholds,
                confidence=dc.confidence, explanation=dc.explanation,
            ))

    # Assign verdicts
    image_verdicts = _assign_verdicts(execution.results, dim_calibrations)

    # Emit per-image verdicts
    if event_bus and has_events:
        for vid, vr in image_verdicts.items():
            event_bus.publish(ImageVerdict(
                module="recalibrator", image_id=vr.image_id,
                image_path=vr.image_path, verdict=vr.verdict,
                scores=vr.scores, errors=[],
            ))

    # Build summary
    methods_used = [dc.method for dc in dim_calibrations.values()]
    total_dims = len(methods_used)
    counts = {}
    for m in methods_used:
        counts[m] = counts.get(m, 0) + 1
    parts = [f"{v}/{total_dims} {k}" for k, v in sorted(counts.items())]
    method_summary = ", ".join(parts)

    overall_confidence = min(
        (dc.confidence for dc in dim_calibrations.values()),
        default=0.0,
    )

    result = RecalibrationResult(
        dimension_calibrations=dim_calibrations,
        image_verdicts=image_verdicts,
        method_summary=method_summary,
        overall_confidence=overall_confidence,
    )

    if event_bus and has_events:
        event_bus.publish(RecalibrationCompleted(
            module="recalibrator",
            method_summary=method_summary,
            overall_confidence=overall_confidence,
        ))

    return result

# Calibration Research — What Works, What Doesn't

> Date: 2026-03-22
> Context: Our pipeline defaults to threshold=0.5 when no exemplars are provided. This research surveys practical approaches to score calibration and threshold selection for image quality assessment.

## The Problem

Our current calibrator uses Platt scaling on good/bad exemplar images. When no exemplars exist (the common case):
- All thresholds default to 0.5
- Blur uses a hardcoded sigmoid centered at Laplacian variance 500
- Exposure uses a hardcoded composite formula
- Content detection passes API confidence through directly
- PixelStats divides by 80 and caps at 1.0 — never calibrated at all

The 0.5 threshold is arbitrary. Depending on the dataset, scores may cluster at 0.2 or 0.9, making 0.5 meaningless.

---

## 1. How Production Systems Handle Thresholds

### LAION Aesthetic Filtering (Stable Diffusion training)
- **Method**: Linear model on CLIP ViT-L/14 embeddings, scores 1–10
- **Threshold calibration**: Visual inspection. Score >= 5 = "looks good." No formal method.
- **Scale**: Filtered 2.37B images
- **Lesson**: Simple works at scale. The model matters more than the threshold method.
- Repo: [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)

### DataComp Competition (ICLR 2024)
- **Winning formula**: CLIP cosine similarity, keep top 30% (threshold 0.243)
- **Threshold**: Empirically chosen, validated on downstream accuracy
- **What failed**: Stronger CLIP models didn't produce better filtering. Text-based filtering didn't help.
- **Key insight**: Filtering methods don't generalize across scales — what works on 64M images may not work on 1B
- Repo: [datacomp](https://github.com/mlfoundations/datacomp)

### Apple Data Filtering Networks (DFN, ICLR 2024)
- **Key finding**: A small model trained on high-quality data makes a better filter than a large accurate model
- **Method**: Train CLIP on ~10M high-quality samples (CC12M), use it to score entire pool, keep top samples
- **Result**: ViT-H on DFN-5B → 84.4% ImageNet zero-shot (best at time of publication)
- Paper: [arxiv.org/abs/2309.17425](https://arxiv.org/abs/2309.17425)

### Cleanlab (Data Quality Scoring)
- **Approach**: Per-class adaptive thresholds = average predicted probability for that class
- **Why it works**: Automatically adjusts for class imbalance and model confidence patterns
- **Requirement**: Out-of-sample predictions via 5-fold cross-validation (never in-sample)
- Repo: [cleanlab](https://github.com/cleanlab/cleanlab)

### CleanVision (Image Quality)
- **Approach**: IQR-based outlier detection for some issues, fixed thresholds for others
- **Issues detected**: blur, dark, light, low information, duplicates, odd aspect ratio
- **No ML models** — CPU-only, statistical methods
- Repo: [cleanvision](https://github.com/cleanlab/cleanvision)

### scikit-learn TunedThresholdClassifierCV (v1.4+)
- **Pattern**: Score everything first, then find optimal threshold via cross-validation
- **Supports**: Any metric (F1, recall, balanced accuracy, custom)
- **Key principle**: Separate the statistical problem (learning probabilities) from the decision problem (setting thresholds)

---

## 2. Unsupervised Threshold Selection Methods

### GMM (Gaussian Mixture Model) — 2-3 Components
- Fit to batch score distribution, find crossing points between Gaussians
- **Pros**: Most principled, gives soft probabilities, adapts to dataset
- **Cons**: Needs 20+ samples, assumes multimodal distribution
- **Gotcha**: Always verify with BIC that 2 components beats 1
- GMM outperforms Otsu: average MSE 257 vs 595

### Multi-Otsu (scikit-image)
- `threshold_multiotsu(scores, classes=3)` — finds 2 thresholds for 3 classes
- Equivalent to 1D k-means, maximizes inter-class variance
- **Pros**: Deterministic, no hyperparameters, fast
- **Cons**: Fails on non-bimodal distributions, sensitive to outliers

### Jenks Natural Breaks
- Finds natural groupings that minimize within-class variance
- **Pros**: Designed for classifying 1D continuous data, fast (C implementation)
- **Cons**: Must specify number of classes a priori
- Use GVF (Goodness of Variance Fit) to evaluate quality — closer to 1.0 = better
- Package: [jenkspy](https://github.com/mthh/jenkspy)

### Hartigan Dip Test
- Tests whether distribution is unimodal (null) vs multimodal
- **Use as prerequisite**: If p < 0.05, distribution is multimodal → GMM/Otsu makes sense. If unimodal → use percentile-based fallback.
- Package: `pip install diptest`

### Adaptive KDE Valley Detection
- Fits kernel density estimate, finds persistent valleys across bandwidths
- Most robust but most complex implementation
- Needs 30+ scores for stable KDE
- Paper: [arxiv.org/abs/2601.14473](https://arxiv.org/abs/2601.14473)

---

## 3. Image Quality Assessment — Benchmark Results

### On Real-World Images (KonIQ-10k, SRCC = Spearman correlation with human ratings)

| Method | KonIQ SRCC | Speed (s/img) | Labels Needed | Notes |
|--------|-----------|---------------|---------------|-------|
| BRISQUE | 0.23 | 0.08 | None | Essentially useless on real images |
| NIQE | 0.38 | 0.07 | None | Also very poor on real-world |
| CLIP-IQA+ | 0.80 | ~0.1 | None | Good, multi-prompt capable |
| QualiCLIP | 0.82 | ~0.1 | None | Zero-shot, no human labels |
| MUSIQ | 0.87 | ~0.06 | Trained | Fast, good accuracy |
| DBCNN | 0.90 | ~0.05 | Trained | Fast, strong |
| TOPIQ-NR | **0.93** | ~0.1 | Trained | Best accuracy at fast speed |
| Q-Align | 0.83 | **15.4** | None | Extremely slow, 7B model |

**Critical finding**: Traditional NR-IQA (BRISQUE, NIQE) are nearly worthless on real-world images. They only work on synthetic distortion datasets. Our OpenCV metrics (Laplacian blur, histogram exposure) fall in this category — they measure real physical properties but the normalization/thresholds are not calibrated to human perception.

**Best bang-for-buck**: TOPIQ-NR (0.93 SRCC at 0.1s/img) if we can accept a trained model. QualiCLIP (0.82 SRCC) if zero-shot is required.

### Comprehensive IQA Library
- **pyiqa** ([IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)): 40+ metrics, unified API (`pyiqa.create_metric('topiq_nr')`)

---

## 4. VLM-Based Quality Assessment

### Pairwise Comparison >> Absolute Scoring
- GenArena (2025): Pairwise VLM comparison achieves **0.86 Spearman** with humans. Absolute scoring: **0.36**. That's a 2.4x improvement.
- **Why**: Relative judgments are much easier for VLMs than absolute ratings
- **Cost**: O(N log N) VLM calls per batch
- Paper: [arxiv.org/abs/2602.06013](https://arxiv.org/abs/2602.06013)

### Q-Align (ICML 2024)
- Rates images as bad/poor/fair/good/excellent, extracts log-probabilities
- Maps naturally to usable/recoverable/unusable
- **Problem**: 15.4 seconds per image on V100. Not viable for pipeline use.
- Repo: [Q-Align](https://github.com/Q-Future/Q-Align)

### EvoQuality (ICLR 2026)
- Self-supervised VLM quality improvement through pairwise voting + GRPO
- Boosts zero-shot PLCC by 31.8% without labels
- Very recent, implementation maturity unclear
- Paper: [arxiv.org/abs/2509.25787](https://arxiv.org/abs/2509.25787)

### GPT-4o for Quality
- **Accuracy**: ~70% on 3-class aesthetic classification (vs 33% chance)
- **Cost**: ~$0.002/image at high detail, $0.0002 at low detail
- **Verdict**: Good for coarse bucketing, poor at fine-grained scoring
- **Best use**: Pairwise comparison prompts, not absolute ratings

---

## 5. Conformal Prediction

### What it offers
- Formal statistical coverage guarantees: "95% of truly usable images will be classified as usable"
- Works with any underlying scorer
- Produces prediction sets (size 0 = reject, 1 = confident, 2 = uncertain)

### Practical implementations
- **MAPIE** ([mapie](https://github.com/scikit-learn-contrib/MAPIE)): sklearn-compatible, 3-line integration
- **RAPS**: On ImageNet, average set size of 2 (vs 19 for basic conformal). As simple as Platt scaling.
- Repo: [conformal_classification](https://github.com/aangelopoulos/conformal_classification)

### Gotchas
- Calibration data must be separate from training data
- Coverage is marginal (averaged) — per-class coverage can vary 92–100%
- Not widely adopted in production yet
- Our 3-class problem (usable/recoverable/unusable) is well-suited — conformal degrades with many classes

---

## 6. Recommended Strategy

### Phase 1 — Batch-Adaptive Calibration (no new dependencies)
Move calibrator after execution. Use the batch's own score distribution:
1. Hartigan dip test to check modality
2. If multimodal: 3-component GMM for natural breaks
3. If unimodal: percentile-based thresholds
4. Report threshold explanations

### Phase 2 — CLIP-IQA Integration
Add QualiCLIP or CLIP-IQA+ as Tier 1 tool. Inherently calibrated [0,1] scores. Replaces/augments OpenCV for quality dimensions.

### Phase 3 — Conformal Prediction Wrapper
Wrap verdict assignment with conformal prediction for formal coverage guarantees. Use MAPIE.

### Phase 4 — VLM Pairwise Calibration (research direction)
For highest quality, use pairwise VLM comparison on a subsample to establish Elo rankings, then map Elo to verdict boundaries.

---

## Key References

- [pyiqa benchmark](https://iqa-pytorch.readthedocs.io/en/latest/benchmark.html) — comprehensive IQA comparison
- [QualiCLIP](https://github.com/miccunifi/QualiCLIP) — zero-shot opinion-free quality scoring
- [CLIP-IQA](https://github.com/IceClear/CLIP-IQA) — multi-prompt quality assessment
- [cleanlab](https://github.com/cleanlab/cleanlab) — per-class adaptive thresholds
- [cleanvision](https://github.com/cleanlab/cleanvision) — image quality detection
- [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) — conformal prediction library
- [jenkspy](https://github.com/mthh/jenkspy) — Jenks natural breaks
- [DataComp](https://github.com/mlfoundations/datacomp) — dataset curation competition
- [Apple DFN](https://arxiv.org/abs/2309.17425) — data filtering networks
- [GenArena](https://arxiv.org/abs/2602.06013) — pairwise VLM quality assessment
- [EvoQuality](https://arxiv.org/abs/2509.25787) — self-evolving VLM quality
- [VisualQuality-R1](https://arxiv.org/abs/2505.14460) — RL-based quality ranking
- [LAION aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [scikit-learn threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)

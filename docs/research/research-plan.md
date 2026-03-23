# Research & Publication Plan

## Target Venue
Workshop paper at **NeurIPS 2026 Data-Centric AI Workshop** or **ICLR 2027 DATA-FM Workshop**

## Research Claim
"LLM-orchestrated multi-tier validation outperforms static pipelines for intent-driven dataset curation."

## Novel Contribution
VLM-based self-calibration for multi-tier validation cascades — deriving quality thresholds from pairwise VLM rankings without exemplar images.

## Experiments Needed

### Experiment A: Quality vs Cost Tradeoff
- Dataset: COCO val2017 (5K images), 5 different intents
- Compare: Tier 1 only / Tier 1+2 / Tier 1+2+3 / all-VLM (no cascade)
- Metric: Precision/recall vs ground-truth labels, cost per image

### Experiment B: Intent Flexibility
- 10 diverse intents across different domains
- Compare: our adaptive pipeline vs fixed-tool pipeline
- Metric: Curation quality variance across intents

### Experiment C: Self-Calibration
- With exemplars / without exemplars / VLM self-calibration (EvoQuality-inspired)
- Metric: Calibration quality (ECE) as function of exemplar count

## Engineering Work Needed
1. Implement VLM self-calibration module (pairwise ranking approach)
2. Add benchmark runner (automated evaluation across datasets/intents)
3. Add ground-truth comparison tool (compare pipeline verdicts vs manual labels)

## Timeline
- Weeks 1-2: VLM self-calibration implementation
- Weeks 2-4: Run experiments A, B, C
- Weeks 4-5: Write paper (6-8 pages)
- Week 6: Submit

## Key References
- EvoQuality (2025) — VLM self-calibration
- ACID/ACED (CVPR 2025) — Active data curation
- Unified Routing & Cascading (ICLR 2025) — Cascade optimization
- DataFlow (2025) — LLM-driven data workflows
- VisualQuality-R1 (NeurIPS 2025) — VLM quality assessment

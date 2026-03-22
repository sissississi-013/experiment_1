# Next Feature: Human-in-the-Loop Calibration

## The Idea

After a validation run completes, the user reviews the results and provides feedback:
- Click an image flagged as "recoverable" → "Actually this is well-lit, the exposure score is wrong"
- Click an image marked "usable" → "This is actually blurry, shouldn't have passed"
- Natural language feedback: "The content detection is too strict — these all clearly have horses"

This feedback is consumed by an agent that adjusts the pipeline's calibration for future runs.

## User Flow (Proposed)

1. Run completes → results shown with verdicts
2. User clicks images and provides feedback (agree/disagree with verdict + optional text)
3. Feedback collected as structured data: `{image_id, original_verdict, user_verdict, feedback_text}`
4. An "adjustment agent" analyzes the feedback and proposes calibration changes
5. Changes are previewed: "Based on your feedback, I'd lower the exposure threshold from 0.45 to 0.35. This would change 3 images from recoverable to usable."
6. User approves → thresholds update for next run

## Open Research Questions

1. **How to translate NL feedback to threshold adjustments?**
   - Option A: Use GPT-4o to interpret feedback and propose specific threshold changes
   - Option B: Treat user verdicts as new exemplars, re-run Platt scaling
   - Option C: Bayesian updating of threshold posteriors from user corrections

2. **How many corrections needed?**
   - Platt scaling needs ~50 samples for reliability (we showed this is a limitation)
   - Could user corrections incrementally improve over multiple runs?
   - EvoQuality shows VLM pairwise ranking can bootstrap without exemplars

3. **Per-run or global calibration?**
   - Should feedback from "horse images" affect "dog images" runs?
   - Probably: per-dimension adjustments are global, per-content adjustments are local

4. **Feedback UI design**
   - Agree/disagree buttons per image (minimum friction)
   - Optional text feedback for nuance
   - Batch feedback: "all of these marked recoverable should be usable"

## Related Research

- EvoQuality (2025): VLM self-calibration through pairwise ranking
- Conformal Prediction: Distribution-free uncertainty bounds
- Active Learning: Select which images to ask the user about (not all of them)
- Apple's Designing Data (2019): Proactive data collection practices

## Prerequisites

- The current scoring improvements (sigmoid blur, composite exposure) should be validated first
- Need to confirm the pipeline produces stable-enough results for feedback to be meaningful
- The persistence layer (Neon) already stores per-image scores — feedback can be stored alongside

## Rough Implementation Sketch

1. **Feedback schema**: `{run_id, image_id, user_verdict, original_verdict, feedback_text, created_at}`
2. **Feedback API**: `POST /api/runs/{id}/feedback` — store corrections
3. **Adjustment agent**: Consumes feedback + original scores, proposes new thresholds
4. **Calibration memory**: Store learned thresholds per dimension, improve over time
5. **Preview mode**: Show "what would change" before applying

## Status

**Not started.** Needs research reading before implementation.
Saved as next major feature after frontend stabilization.

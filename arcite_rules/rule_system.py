"""
ARCITE Rule Generation System
==============================
GenAI@Berkeley × Apple

This module handles everything related to rule-based image validation:

1. Model Registry   — metadata for every available CV model
2. Rule Schema      — the structured ValidationRule format
3. Rule Generator   — LLM-based rule generation from user intent
4. Storage helpers  — save/load/query rules (file-based, Neon upgrade path)
5. Conversion       — converts RuleSet → Andrew's ValidationContract / BasePayload
6. Human-in-loop    — threshold adjustment + confirmation helpers

Andrew's team calls generate_rules(user_intent) and gets back a list of
ValidationRule objects they apply deterministically — zero LLM at inference time.
"""

import json
import uuid
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict


# ─────────────────────────────────────────────────────────────
# 1. MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
# Every model the pipeline can call. The LLM reads this to
# generate rules with valid thresholds and operators.
# To add a new model: call register_model() with a ModelMetadata.
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelMetadata:
    """Describes one CV model available in the tool registry."""
    model_id: str           # Unique key, e.g. "laplacian_blur"
    model_name: str         # Human-readable name
    description: str        # What it does (shown to the LLM)
    output_type: str        # "float", "int", "boolean", "classification", "detection"
    output_range: list      # [min, max] for numeric, or list of possible labels
    output_unit: str        # What the number means, e.g. "variance (higher=sharper)"
    tier: int               # 1=CPU(<1ms), 2=API(~500ms), 3=VLM(~2s)
    example_output: Any     # A concrete example so the LLM understands the format
    default_operator: str   # Suggested operator for typical use
    default_threshold: float  # Reasonable default threshold
    notes: str = ""         # Any caveats


MODEL_REGISTRY: Dict[str, ModelMetadata] = {}


def register_model(m: ModelMetadata):
    MODEL_REGISTRY[m.model_id] = m


# ── Tier 1: CPU tools (<1ms) ──────────────────────────────

register_model(ModelMetadata(
    model_id="laplacian_blur",
    model_name="Laplacian Blur Detector",
    description=(
        "Measures image sharpness using the variance of the Laplacian filter. "
        "Higher values = sharper image. Low values indicate blur."
    ),
    output_type="float",
    output_range=[0, 2000],
    output_unit="variance (higher = sharper)",
    tier=1,
    example_output=142.7,
    default_operator=">",
    default_threshold=100.0,
    notes="Values below 50 are very blurry. Sharp photos are typically 200+.",
))

register_model(ModelMetadata(
    model_id="histogram_exposure",
    model_name="Histogram Exposure Analyzer",
    description=(
        "Analyzes the brightness distribution of an image. Returns a score "
        "from 0 to 1 where 0.5 is ideal exposure. Values near 0 = underexposed, "
        "near 1 = overexposed."
    ),
    output_type="float",
    output_range=[0.0, 1.0],
    output_unit="exposure score (0.5 = ideal)",
    tier=1,
    example_output=0.47,
    default_operator=">=",
    default_threshold=0.2,
    notes="Typically used with a range: reject if < 0.15 or > 0.85.",
))

register_model(ModelMetadata(
    model_id="pixel_stats",
    model_name="Pixel Statistics Calculator",
    description=(
        "Returns basic image statistics: mean pixel intensity (0-255). "
        "Useful for detecting fully black/white or very dark/bright images."
    ),
    output_type="float",
    output_range=[0, 255],
    output_unit="mean pixel intensity (0=black, 255=white)",
    tier=1,
    example_output=127.3,
    default_operator=">",
    default_threshold=20.0,
    notes="Images with mean < 10 are essentially black. Mean > 245 essentially white.",
))

register_model(ModelMetadata(
    model_id="resolution_check",
    model_name="Resolution Checker",
    description=(
        "Returns the smaller dimension (width or height) in pixels. "
        "Use to filter out images that are too small for training."
    ),
    output_type="int",
    output_range=[1, 10000],
    output_unit="pixels (minimum of width, height)",
    tier=1,
    example_output=1024,
    default_operator=">=",
    default_threshold=256,
    notes="Most training pipelines want at least 224px. High-quality datasets use 512+.",
))

register_model(ModelMetadata(
    model_id="aspect_ratio",
    model_name="Aspect Ratio Calculator",
    description=(
        "Returns the aspect ratio as width/height. Useful for filtering "
        "extremely elongated or narrow images."
    ),
    output_type="float",
    output_range=[0.1, 10.0],
    output_unit="ratio (1.0 = square, >1 = landscape, <1 = portrait)",
    tier=1,
    example_output=1.33,
    default_operator=">=",
    default_threshold=0.5,
    notes="Typical photos range from 0.56 (9:16 portrait) to 1.78 (16:9 landscape).",
))

# ── Tier 2: API tools (~500ms) ────────────────────────────

register_model(ModelMetadata(
    model_id="grounding_dino",
    model_name="NVIDIA GroundingDINO Object Detector",
    description=(
        "Open-vocabulary object detection. Given a text prompt (target), "
        "returns a confidence score (0-1) for whether that object is present "
        "in the image. Can detect any object described in natural language."
    ),
    output_type="float",
    output_range=[0.0, 1.0],
    output_unit="detection confidence (higher = more confident)",
    tier=2,
    example_output=0.91,
    default_operator=">",
    default_threshold=0.5,
    notes=(
        "Scores above 0.7 are high confidence. Below 0.3 is weak/absent. "
        "Works for any noun: 'horse', 'red car', 'person wearing hat'."
    ),
))

register_model(ModelMetadata(
    model_id="roboflow_classifier",
    model_name="Roboflow Image Classifier",
    description=(
        "Image classification using a pre-trained or custom Roboflow model. "
        "Returns a confidence score (0-1) for whether the image belongs to "
        "a specific class."
    ),
    output_type="float",
    output_range=[0.0, 1.0],
    output_unit="classification confidence (higher = more likely)",
    tier=2,
    example_output=0.88,
    default_operator=">",
    default_threshold=0.7,
    notes="Custom models may have different class labels. Check model docs.",
))

register_model(ModelMetadata(
    model_id="nsfw_detector",
    model_name="NSFW Content Detector",
    description=(
        "Detects inappropriate or explicit content in images. Returns "
        "a score from 0 to 1 where higher values indicate more explicit content."
    ),
    output_type="float",
    output_range=[0.0, 1.0],
    output_unit="NSFW probability (higher = more explicit)",
    tier=2,
    example_output=0.03,
    default_operator="<",
    default_threshold=0.3,
    notes="Safe images typically score < 0.1. Threshold depends on use case strictness.",
))

register_model(ModelMetadata(
    model_id="watermark_detector",
    model_name="Watermark Detector",
    description=(
        "Detects watermarks, logos, or overlaid text in images. Returns "
        "a confidence score (0-1) for watermark presence."
    ),
    output_type="float",
    output_range=[0.0, 1.0],
    output_unit="watermark probability (higher = more likely)",
    tier=2,
    example_output=0.12,
    default_operator="<",
    default_threshold=0.5,
    notes="Stock photo watermarks typically score 0.7+. Subtle logos may be 0.3-0.5.",
))

# ── Tier 3: VLM tools (~2s) ──────────────────────────────

register_model(ModelMetadata(
    model_id="gpt4o_vision",
    model_name="GPT-4o Vision (Semantic Quality)",
    description=(
        "Uses GPT-4o to semantically evaluate image content. Can assess "
        "subjective qualities like 'artistic quality', 'relevance to prompt', "
        "or 'scene complexity'. Returns a score from 0 to 1."
    ),
    output_type="float",
    output_range=[0.0, 1.0],
    output_unit="semantic relevance score (higher = more relevant)",
    tier=3,
    example_output=0.82,
    default_operator=">",
    default_threshold=0.6,
    notes=(
        "Expensive and slow. Use as fallback when cheaper models are insufficient. "
        "Score meaning depends on the prompt given to the VLM."
    ),
))


# ─────────────────────────────────────────────────────────────
# 2. RULE SCHEMA
# ─────────────────────────────────────────────────────────────

@dataclass
class ValidationRule:
    """
    One atomic rule that maps a model output to a pass/fail decision.

    Applied deterministically:
        model_output = run_model(image, model_id)
        passed = OPERATORS[operator](model_output, threshold)

    No LLM at inference time — pure numeric comparison.
    """
    rule_id: str        # Human-readable ID, e.g. "blur_check"
    model_id: str       # Which model to run; must exist in MODEL_REGISTRY
    target: str         # What to look for: "horse", "blur", "exposure"
    threshold: float    # The numeric cutoff
    operator: str       # ">", ">=", "<", "<=", "=="

    def to_dict(self) -> dict:
        return asdict(self)

    def explain(self) -> str:
        model = MODEL_REGISTRY.get(self.model_id)
        model_name = model.model_name if model else self.model_id
        return (
            f"Rule '{self.rule_id}': Run {model_name} looking for '{self.target}'. "
            f"Image passes if output {self.operator} {self.threshold}."
        )


# Operator functions for deterministic evaluation
OPERATORS = {
    ">":  lambda x, y: x > y,
    ">=": lambda x, y: x >= y,
    "<":  lambda x, y: x < y,
    "<=": lambda x, y: x <= y,
    "==": lambda x, y: x == y,
}


def evaluate_rule(rule: ValidationRule, model_output: float) -> bool:
    """Apply a rule to a model output. Returns True if the image passes."""
    return OPERATORS[rule.operator](model_output, rule.threshold)


# ─────────────────────────────────────────────────────────────
# 3. RULE SET
# ─────────────────────────────────────────────────────────────

@dataclass
class RuleSet:
    """A collection of rules generated for one validation run."""
    run_id: str
    created_at: str
    user_intent: str                        # Original natural language request
    rules: List[ValidationRule]
    confirmed: bool = False                 # Has a human approved these rules?
    notes: str = ""                         # Notes from the human reviewer
    conversation_history: List[dict] = field(default_factory=list)  # Full chat context

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "user_intent": self.user_intent,
            "rules": [r.to_dict() for r in self.rules],
            "confirmed": self.confirmed,
            "notes": self.notes,
            "conversation_history": self.conversation_history,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ─────────────────────────────────────────────────────────────
# 4. RULE GENERATOR — LLM prompt + parsing
# ─────────────────────────────────────────────────────────────

def _build_model_registry_context() -> str:
    """Format the model registry as readable context for the LLM prompt."""
    lines = []
    for m in MODEL_REGISTRY.values():
        tier_label = "CPU <1ms" if m.tier == 1 else "API ~500ms" if m.tier == 2 else "VLM ~2s"
        lines.append(
            f"Model: {m.model_id}\n"
            f"  Name: {m.model_name}\n"
            f"  Description: {m.description}\n"
            f"  Output type: {m.output_type}\n"
            f"  Output range: {m.output_range}\n"
            f"  Output unit: {m.output_unit}\n"
            f"  Tier: {m.tier} ({tier_label})\n"
            f"  Example output: {m.example_output}\n"
            f"  Default operator: {m.default_operator}\n"
            f"  Default threshold: {m.default_threshold}\n"
            f"  Notes: {m.notes}\n"
        )
    return "\n".join(lines)


_SYSTEM_PROMPT = """You are a rule generation engine for a computer vision dataset validation pipeline.

Your job: take a user's natural language description of what images they want, and produce structured validation rules that can be applied DETERMINISTICALLY to filter images. No AI is used at inference time — only numeric comparisons.

## Available Models

These are the ONLY models you can reference. Each model produces a single numeric output. You must set thresholds within the model's output range.

{model_registry}

## Output Format

Respond with ONLY a JSON array of rule objects. No explanation, no markdown, no backticks. Each rule has exactly these fields:

[
  {{
    "rule_id": "string — descriptive snake_case id like 'blur_check' or 'horse_detect'",
    "model_id": "string — must be one of the model IDs listed above",
    "target": "string — what the rule is checking for, e.g. 'horse', 'blur', 'exposure'",
    "threshold": number,
    "operator": "string — one of: >, >=, <, <=, =="
  }}
]

## Rules for Rule Generation

1. Generate EXACTLY ONE rule per model you select. Do not generate multiple rules for the same model.
2. Only select models RELEVANT to what the user asked for. Do not include unnecessary models.
3. Thresholds MUST fall within the model's output range. Check the range before setting a threshold.
4. Choose the operator that matches the model's semantics:
   - For quality scores where higher is better (blur, resolution): use > or >=
   - For probability of unwanted content (NSFW, watermark): use < or <=
5. ALWAYS include basic quality rules (blur, exposure) unless the user explicitly says not to.
6. Prefer cheaper models (lower tier) when multiple models could serve the same purpose.
7. Adjust thresholds based on how strict the user's language is:
   - "high quality", "strict", "only the best" → stricter thresholds
   - "any", "decent", "okay" → more lenient thresholds
   - No qualifier → use the default threshold
8. The rule_id should be descriptive: 'detect_horse' not 'rule_1'.
"""

_USER_PROMPT_TEMPLATE = """Generate validation rules for this request:

"{user_intent}"

Remember: respond with ONLY the JSON array, nothing else."""


def _parse_rules_response(response_text: str) -> List[ValidationRule]:
    """Parse the LLM's JSON response into ValidationRule objects."""
    text = response_text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    raw_rules = json.loads(text)
    rules = []

    for r in raw_rules:
        if r["model_id"] not in MODEL_REGISTRY:
            print(f"WARNING: Unknown model_id '{r['model_id']}', skipping rule '{r.get('rule_id', '?')}'")
            continue

        if r["operator"] not in OPERATORS:
            print(f"WARNING: Invalid operator '{r['operator']}', skipping rule '{r.get('rule_id', '?')}'")
            continue

        # Clamp threshold to model's output range
        model = MODEL_REGISTRY[r["model_id"]]
        lo, hi = model.output_range[0], model.output_range[1]
        if not (lo <= r["threshold"] <= hi):
            print(
                f"WARNING: Threshold {r['threshold']} outside range [{lo}, {hi}] "
                f"for '{r['model_id']}'. Clamping."
            )
            r["threshold"] = max(lo, min(hi, r["threshold"]))

        rules.append(ValidationRule(
            rule_id=r["rule_id"],
            model_id=r["model_id"],
            target=r["target"],
            threshold=r["threshold"],
            operator=r["operator"],
        ))

    return rules


def generate_rules(user_intent: str, llm_call_fn=None) -> Optional["RuleSet"]:
    """
    End-to-end rule generation.

    Args:
        user_intent:  Natural language description of what images the user wants.
        llm_call_fn:  Callable(system_prompt: str, user_prompt: str) -> str.
                      If None, prints the prompts for manual use and returns None.
                      Use make_openai_llm_fn() from openai_llm.py to wire up OpenAI.

    Returns:
        A RuleSet with confirmed=False, ready for human review.
    """
    system_prompt = _SYSTEM_PROMPT.format(model_registry=_build_model_registry_context())
    user_prompt = _USER_PROMPT_TEMPLATE.format(user_intent=user_intent)

    if llm_call_fn is None:
        print("=" * 60)
        print("SYSTEM PROMPT:")
        print("=" * 60)
        print(system_prompt)
        print()
        print("=" * 60)
        print("USER PROMPT:")
        print("=" * 60)
        print(user_prompt)
        print()
        print("Paste the LLM's response into _parse_rules_response() to continue.")
        return None

    response_text = llm_call_fn(system_prompt, user_prompt)
    rules = _parse_rules_response(response_text)

    return RuleSet(
        run_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        user_intent=user_intent,
        rules=rules,
        confirmed=False,
        conversation_history=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response_text},
        ],
    )


# ─────────────────────────────────────────────────────────────
# 5. STORAGE — save/load/query rules
# ─────────────────────────────────────────────────────────────
# File-based for now. Swap in Neon Postgres later by replacing
# the implementation of save_ruleset / load_ruleset.
# ─────────────────────────────────────────────────────────────

_DEFAULT_STORAGE_DIR = os.path.join(os.path.dirname(__file__), "rule_storage")


def save_ruleset(rule_set: RuleSet, directory: str = _DEFAULT_STORAGE_DIR) -> str:
    """Save a RuleSet to disk as JSON. Returns the file path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{rule_set.run_id}.json")
    with open(path, "w") as f:
        f.write(rule_set.to_json())
    return path


def load_ruleset(run_id: str, directory: str = _DEFAULT_STORAGE_DIR) -> Optional[RuleSet]:
    """Load a RuleSet by run_id. Returns None if not found."""
    path = os.path.join(directory, f"{run_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return RuleSet(
        run_id=data["run_id"],
        created_at=data["created_at"],
        user_intent=data["user_intent"],
        rules=[ValidationRule(**r) for r in data["rules"]],
        confirmed=data.get("confirmed", False),
        notes=data.get("notes", ""),
        conversation_history=data.get("conversation_history", []),
    )


def list_rulesets(directory: str = _DEFAULT_STORAGE_DIR) -> List[dict]:
    """List all saved RuleSets (summary only, no full rule data)."""
    if not os.path.exists(directory):
        return []
    results = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                data = json.load(f)
            results.append({
                "run_id": data["run_id"],
                "created_at": data["created_at"],
                "user_intent": data["user_intent"],
                "num_rules": len(data["rules"]),
                "confirmed": data.get("confirmed", False),
            })
    return results


def find_rules_by_model(model_id: str, directory: str = _DEFAULT_STORAGE_DIR) -> List[dict]:
    """Find all rules across all runs that use a specific model."""
    results = []
    if not os.path.exists(directory):
        return results
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                data = json.load(f)
            for rule in data["rules"]:
                if rule["model_id"] == model_id:
                    results.append({"run_id": data["run_id"], "user_intent": data["user_intent"], **rule})
    return results


# ─────────────────────────────────────────────────────────────
# 6. CONVERSION TO ANDREW'S SCHEMA
# ─────────────────────────────────────────────────────────────

def ruleset_to_validation_contracts(rule_set: RuleSet) -> List[dict]:
    """
    Convert a RuleSet to Andrew's ValidationContract format.

    Groups our rules by model_id (one contract per model).
    LogicGateway include/exclude/final fields are left None —
    Andrew's team fills these in based on their evaluation logic.
    """
    by_model: Dict[str, List[ValidationRule]] = {}
    for rule in rule_set.rules:
        by_model.setdefault(rule.model_id, []).append(rule)

    contracts = []
    for model_id, rules in by_model.items():
        contracts.append({
            "model_id": model_id,
            "expected_targets": list({r.target for r in rules}),
            "rules": [
                {
                    "id": r.rule_id,
                    "targets": [r.target],
                    "thresholds": [r.threshold],
                    "operator": r.operator,
                }
                for r in rules
            ],
            "logic_gateway": {
                "include": None,
                "exclude": None,
                "final": None,
            },
        })
    return contracts


def ruleset_to_base_payload(
    rule_set: RuleSet,
    job_id: str,
    database_id: str,
    image_name: str,
    collection_id: str,
) -> dict:
    """
    Build a complete BasePayload in Andrew's format.

    model_executions is left empty — his pipeline fills it after running models.
    The conversation_history from rule generation is included in provenance.
    """
    return {
        "request_metadata": {
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "data_source": {
            "database_id": database_id,
            "image_name": image_name,
            "collection_id": collection_id,
        },
        "model_executions": [],  # Andrew's pipeline fills this
        "validation_contracts": ruleset_to_validation_contracts(rule_set),
    }


def ruleset_to_manifest_provenance(rule_set: RuleSet) -> dict:
    """
    Build the provenance block for the Global Manifest Schema.

    Includes the full conversation history and the compiled contract,
    matching the manifest format from the design doc.
    """
    return {
        "llm_orchestrator_model": "gpt-4o",
        "user_intent_history": rule_set.conversation_history,
        "compiled_contract": {
            "rules": [
                {
                    "id": r.rule_id,
                    "target": r.target,
                    "operator": r.operator,
                    "threshold": r.threshold,
                }
                for r in rule_set.rules
            ]
        },
    }


# ─────────────────────────────────────────────────────────────
# 7. HUMAN-IN-THE-LOOP HELPERS
# ─────────────────────────────────────────────────────────────

def adjust_threshold(rule_set: RuleSet, rule_id: str, new_threshold: float) -> bool:
    """Let a human adjust a threshold before confirming the rule set."""
    for rule in rule_set.rules:
        if rule.rule_id == rule_id:
            model = MODEL_REGISTRY.get(rule.model_id)
            if model:
                lo, hi = model.output_range
                if not (lo <= new_threshold <= hi):
                    print(f"Threshold {new_threshold} is outside range [{lo}, {hi}]")
                    return False
            rule.threshold = new_threshold
            return True
    print(f"Rule '{rule_id}' not found")
    return False


def confirm_ruleset(rule_set: RuleSet, notes: str = "") -> RuleSet:
    """Mark a rule set as human-approved."""
    rule_set.confirmed = True
    rule_set.notes = notes
    return rule_set


def display_ruleset(rule_set: RuleSet):
    """Pretty-print a rule set for human review."""
    print(f"\n{'='*60}")
    print(f"Rule Set: {rule_set.run_id}")
    print(f"Created:  {rule_set.created_at}")
    print(f'Intent:   "{rule_set.user_intent}"')
    print(f"Status:   {'✅ CONFIRMED' if rule_set.confirmed else '⏳ PENDING REVIEW'}")
    print(f"{'='*60}")
    for i, rule in enumerate(rule_set.rules, 1):
        model = MODEL_REGISTRY.get(rule.model_id)
        tier_label = f"T{model.tier}" if model else "?"
        range_label = f"[{model.output_range[0]}, {model.output_range[1]}]" if model else "?"
        print(f"\n  Rule {i}: {rule.rule_id}")
        print(f"    Model:     {rule.model_id} ({tier_label})")
        print(f"    Target:    {rule.target}")
        print(f"    Condition: output {rule.operator} {rule.threshold}")
        print(f"    Range:     {range_label}")
        print(f"    Meaning:   {rule.explain()}")
    if rule_set.notes:
        print(f"\n  Notes: {rule_set.notes}")
    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# 8. DEMO (no LLM required)
# ─────────────────────────────────────────────────────────────

def demo_without_llm():
    """Full demo using hardcoded rules (simulates LLM output)."""
    print("\n" + "="*60)
    print("ARCITE RULE SYSTEM — DEMO")
    print("="*60)

    user_intent = (
        "I want high-quality images of horses for training an RL model. "
        "No cars, no blur, no watermarks."
    )

    # Simulate what the LLM would return
    simulated_llm_response = json.dumps([
        {"rule_id": "sharpness_check",   "model_id": "laplacian_blur",        "target": "blur",       "threshold": 150.0, "operator": ">"},
        {"rule_id": "exposure_check",    "model_id": "histogram_exposure",    "target": "exposure",   "threshold": 0.15,  "operator": ">="},
        {"rule_id": "min_resolution",    "model_id": "resolution_check",      "target": "resolution", "threshold": 512,   "operator": ">="},
        {"rule_id": "detect_horse",      "model_id": "grounding_dino",        "target": "horse",      "threshold": 0.7,   "operator": ">"},
        {"rule_id": "exclude_cars",      "model_id": "roboflow_classifier",   "target": "car",        "threshold": 0.5,   "operator": "<"},
        {"rule_id": "no_watermarks",     "model_id": "watermark_detector",    "target": "watermark",  "threshold": 0.4,   "operator": "<"},
    ])

    rules = _parse_rules_response(simulated_llm_response)
    rule_set = RuleSet(
        run_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        user_intent=user_intent,
        rules=rules,
    )

    display_ruleset(rule_set)

    # Human adjusts blur threshold
    print("Human adjusts blur threshold: 150 → 200 (stricter)")
    adjust_threshold(rule_set, "sharpness_check", 200.0)

    confirm_ruleset(rule_set, notes="Approved after tightening blur threshold.")
    display_ruleset(rule_set)

    path = save_ruleset(rule_set)
    print(f"Saved to: {path}")

    # Convert to Andrew's format
    contracts = ruleset_to_validation_contracts(rule_set)
    print("\nValidationContracts (Andrew's format):")
    print(json.dumps(contracts, indent=2))

    payload = ruleset_to_base_payload(
        rule_set,
        job_id=str(uuid.uuid4()),
        database_id="apple_vision_db",
        image_name="horse_001.jpg",
        collection_id="training_v1",
    )
    print("\nBasePayload:")
    print(json.dumps(payload, indent=2))

    # Manifest provenance block
    provenance = ruleset_to_manifest_provenance(rule_set)
    print("\nManifest Provenance Block:")
    print(json.dumps(provenance, indent=2))

    # Simulated evaluation
    print("\n" + "="*60)
    print("SIMULATED EVALUATION — horse_001.jpg")
    print("="*60)
    fake_outputs = {
        "laplacian_blur":      245.3,
        "histogram_exposure":  0.48,
        "resolution_check":    1024,
        "grounding_dino":      0.91,
        "roboflow_classifier": 0.12,
        "watermark_detector":  0.05,
    }
    print(f"\nModel outputs: {json.dumps(fake_outputs, indent=2)}\n")
    all_passed = True
    for rule in rule_set.rules:
        output = fake_outputs.get(rule.model_id, 0.0)
        passed = evaluate_rule(rule, output)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {rule.rule_id}: {output} {rule.operator} {rule.threshold} → {status}")
        if not passed:
            all_passed = False
    print(f"\n  Final verdict: {'✅ IMAGE APPROVED' if all_passed else '❌ IMAGE REJECTED'}")

    return rule_set


if __name__ == "__main__":
    demo_without_llm()

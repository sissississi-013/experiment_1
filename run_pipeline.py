#!/usr/bin/env python3
"""
Run the validation pipeline interactively and print the full report.

Usage:
    python3 run_pipeline.py "find 10 sharp, well-exposed images of horses"
    python3 run_pipeline.py --dataset-path /path/to/images "find sharp images"
"""
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

from validation_pipeline.config import PipelineConfig
from validation_pipeline.schemas.user_input import UserInput
from validation_pipeline.pipeline import ValidationPipeline
from validation_pipeline.event_bus import EventBus
from validation_pipeline.events import (
    PipelineEvent, ModuleStarted, ModuleCompleted, ImageProgress,
    ImageVerdict, SpecGenerated, PlanGenerated, DatasetResolved,
    PipelineErrorEvent, ToolProgress,
)


def print_header(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_section(text):
    print(f"\n--- {text} ---")


def cli_subscriber(event: PipelineEvent):
    """Print real-time progress to console."""
    if isinstance(event, ModuleStarted):
        detail = f" — {event.details}" if event.details else ""
        print(f"  [{event.module}] Started{detail}")
    elif isinstance(event, ModuleCompleted):
        summary = f" — {event.summary}" if event.summary else ""
        print(f"  [{event.module}] Completed ({event.duration_seconds:.1f}s){summary}")
    elif isinstance(event, DatasetResolved):
        print(f"  [dataset_resolver] Downloaded {event.image_count} images from {event.source}")
    elif isinstance(event, SpecGenerated):
        print(f"  [spec_generator] Spec: {event.spec_summary}")
    elif isinstance(event, PlanGenerated):
        print(f"  [planner] Plan: {event.steps_count} steps across tiers {event.tiers}")
    elif isinstance(event, ImageProgress):
        print(f"  [executor] {event.current}/{event.total} {os.path.basename(event.image_path)}", end="", flush=True)
    elif isinstance(event, ImageVerdict):
        scores = ", ".join(f"{k}={v:.2f}" for k, v in event.scores.items())
        icon = {"usable": "O", "recoverable": "~", "unusable": "X", "error": "!"}
        print(f" [{icon.get(event.verdict, '?')}] {event.verdict} ({scores})")
    elif isinstance(event, PipelineErrorEvent):
        print(f"  [{event.module}] ERROR: {event.message}")


def run(intent: str, dataset_path: str | None = None, dataset_description: str | None = None):
    config = PipelineConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    bus = EventBus()
    bus.subscribe_all(cli_subscriber)
    pipeline = ValidationPipeline(config, event_bus=bus)

    user_input = UserInput(
        intent=intent,
        dataset_path=dataset_path,
        dataset_description=dataset_description,
    )

    print_header("VALIDATION PIPELINE")
    print(f"Intent: {intent}")
    if dataset_path:
        print(f"Dataset: {dataset_path}")
    if dataset_description:
        print(f"Dataset description: {dataset_description}")

    report = pipeline.run(user_input, auto_approve=True)

    # Dataset stats
    print_header("DATASET STATS")
    stats = report.dataset_stats
    print(f"Total images:  {stats.total_images}")
    print(f"Usable:        {stats.usable} ({stats.usable_percentage:.1f}%)")
    print(f"Recoverable:   {stats.recoverable}")
    print(f"Unusable:      {stats.unusable}")
    if stats.flag_breakdown:
        print(f"Flag breakdown: {stats.flag_breakdown}")

    # Curation score
    print_header("CURATION SCORE")
    score = report.curation_score
    print(f"Overall: {score.overall_score:.2f} (confidence: {score.confidence:.2f})")
    print(f"Explanation: {score.explanation}")
    if score.dimension_scores:
        print("Per-dimension scores:")
        for dim, s in score.dimension_scores.items():
            print(f"  {dim}: {s:.2f}")

    # Per-image results
    print_header("PER-IMAGE RESULTS")
    for img in report.per_image_results:
        verdict_icon = {"usable": "O", "recoverable": "~", "unusable": "X"}
        icon = verdict_icon.get(img.verdict, "?")
        name = os.path.basename(img.image_path)
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in img.scores.items())
        print(f"  [{icon}] {name:30s} {img.verdict:12s} {scores_str}")
        if img.flags:
            print(f"      flags: {', '.join(img.flags)}")
        if img.recovery_suggestion:
            print(f"      recovery: {img.recovery_suggestion}")
        if img.audit:
            for line in img.audit:
                status = "PASS" if line.passed else "FAIL"
                print(f"      L{line.line_number}: {line.tool_name} raw={line.raw_value} "
                      f"cal={line.calibrated_value:.2f} thr={line.threshold:.2f} [{status}]")

    # Audit trail summary
    print_header("AUDIT TRAIL")
    trail = report.audit_trail
    print(f"LLM model: {trail.llm_model_used}")
    print(f"Timestamp: {trail.timestamp}")
    print(f"Tools used: {list(trail.tool_versions.keys())}")

    # Plan summary from audit
    if "steps" in trail.plan:
        print_section("Plan steps")
        for step in trail.plan["steps"]:
            tier = step.get("tier", "?")
            tool = step.get("tool_name", "?")
            dim = step.get("dimension", "?")
            thr = step.get("threshold", "?")
            params = step.get("tool_params", None)
            print(f"  Tier {tier}: {tool} -> {dim} (threshold={thr})", end="")
            if params:
                print(f" params={params}", end="")
            print()

    print_header("DONE")
    print(f"Report ID: {report.report_id}")

    # Save full report to JSON
    out_path = "output/report.json"
    os.makedirs("output", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"Full report saved to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_pipeline.py <intent> [--dataset-path PATH] [--dataset-desc DESCRIPTION]")
        print()
        print("Examples:")
        print('  python3 run_pipeline.py "find 10 sharp images of horses from COCO"')
        print('  python3 run_pipeline.py "find sharp images" --dataset-path /path/to/images')
        print('  python3 run_pipeline.py "find well-lit cat photos" --dataset-desc "20 cat images from COCO val2017"')
        sys.exit(1)

    intent = sys.argv[1]
    dataset_path = None
    dataset_description = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--dataset-path" and i + 1 < len(sys.argv):
            dataset_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--dataset-desc" and i + 1 < len(sys.argv):
            dataset_description = sys.argv[i + 1]
            i += 2
        else:
            # Treat remaining args as part of intent if no flag
            intent += " " + sys.argv[i]
            i += 1

    # If no dataset_path and no dataset_desc, extract from intent
    if not dataset_path and not dataset_description:
        dataset_description = intent

    run(intent, dataset_path, dataset_description)

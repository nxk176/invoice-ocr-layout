"""Benchmark orchestration and N/A-aware report generation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from invoice_ocr.evaluation.aggregate import evaluate_run
from invoice_ocr.pipeline import PipelineSelection, enumerate_pipeline_combinations
from invoice_ocr.training.datasets import validate_ground_truth


@dataclass
class MetricAvailability:
    value: float | None
    status: str
    reason: str | None = None


def stage_metric_availability(gt_root: Path) -> dict[str, MetricAvailability]:
    report = validate_ground_truth(gt_root)
    return {
        "detection": MetricAvailability(
            None,
            "pending" if report.detection_count else "N/A",
            None if report.detection_count else "GT/detection annotations are missing",
        ),
        "recognition": MetricAvailability(
            None,
            "pending" if report.recognition_count else "N/A",
            None if report.recognition_count else "GT/recognition annotations are missing",
        ),
        "layout": MetricAvailability(
            None,
            "pending" if report.layout_count else "N/A",
            None if report.layout_count else "GT/layout annotations are missing",
        ),
        "final": MetricAvailability(
            None,
            "pending" if report.final_count else "N/A",
            None if report.final_count else "GT/final annotations are missing",
        ),
    }


def write_benchmark_reports(
    output_dir: Path,
    rows: list[dict[str, Any]],
    gt_root: Path,
    *,
    data_root: Path | None = None,
    gt_prefix: str | None = None,
    target_manifest: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        run_name = f"{row['detector']}__{row['recognizer']}__{row['layout']}"
        row.update(
            evaluate_run(
                output_dir / run_name,
                gt_root,
                data_root,
                gt_prefix,
                target_manifest,
            )
        )
    availability = stage_metric_availability(gt_root)
    metrics = {
        "pipeline_count": len(rows),
        "stage_metric_availability": {key: asdict(value) for key, value in availability.items()},
        "pipelines": rows,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = sorted({key for row in rows for key in row}) or [
        "detector",
        "recognizer",
        "layout",
    ]
    for filename in ("metrics.csv", "comparison.csv"):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    summary_lines = [
        "# Benchmark summary",
        "",
        f"- Pipeline combinations: {len(rows)}",
    ]
    for stage, state in availability.items():
        suffix = f" — {state.reason}" if state.reason else ""
        summary_lines.append(f"- {stage}: {state.status}{suffix}")
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def benchmark_rows(all_combinations: bool) -> list[dict[str, str]]:
    selections: list[PipelineSelection] = (
        enumerate_pipeline_combinations() if all_combinations else []
    )
    return [
        {
            "detector": selection.detector,
            "recognizer": selection.recognizer,
            "layout": selection.layout,
            "status": "not_run",
        }
        for selection in selections
    ]

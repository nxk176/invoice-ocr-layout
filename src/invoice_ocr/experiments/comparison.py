"""Fair before/after comparison with metric-aware improvement direction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from invoice_ocr.experiments.hashing import canonical_json_hash

LOWER_IS_BETTER_TOKENS = (
    "cer",
    "wer",
    "latency",
    "time",
    "runtime",
    "memory",
    "ram",
    "unresolved",
    "failed",
)
HIGHER_IS_BETTER_TOKENS = (
    "precision",
    "recall",
    "f1",
    "hmean",
    "iou",
    "exact",
    "accuracy",
    "throughput",
    "success",
)


def lower_is_better(metric_name: str, metric: dict[str, Any] | None = None) -> bool:
    if metric is not None and isinstance(metric.get("lower_is_better"), bool):
        return bool(metric["lower_is_better"])
    normalized = metric_name.casefold()
    if any(token in normalized for token in LOWER_IS_BETTER_TOKENS):
        return True
    if any(token in normalized for token in HIGHER_IS_BETTER_TOKENS):
        return False
    return False


def absolute_change(before: float, after: float) -> float:
    return after - before


def relative_change_percent(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return (after - before) / abs(before) * 100


def classify_change(metric_name: str, before: float, after: float) -> str:
    if after == before:
        return "unchanged"
    improved = after < before if lower_is_better(metric_name) else after > before
    return "improved" if improved else "regressed"


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required run artifact not found: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _metric_entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for name, raw in metrics.items():
        if isinstance(raw, dict):
            entries[str(name)] = raw
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            entries[str(name)] = {
                "value": float(raw),
                "lower_is_better": lower_is_better(str(name)),
            }
    return entries


def _numeric_values(payload: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, entry in _metric_entries(payload).items():
        value = entry.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = float(value)
    return result


def _timing_values(payload: dict[str, Any]) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in payload.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _fairness_checks(
    before: dict[str, Any],
    after: dict[str, Any],
    allow_incomparable: bool,
) -> tuple[bool, list[str]]:
    checks = {
        "split_manifest_hash": (
            before.get("split_manifest_hash"),
            after.get("split_manifest_hash"),
        ),
        "test_document_ids": (
            sorted(before.get("test_document_ids", [])),
            sorted(after.get("test_document_ids", [])),
        ),
        "schema_version": (before.get("schema_version"), after.get("schema_version")),
        "preprocessing_config_hash": (
            before.get("preprocessing_config_hash"),
            after.get("preprocessing_config_hash"),
        ),
        "postprocessing_config_hash": (
            before.get("postprocessing_config_hash"),
            after.get("postprocessing_config_hash"),
        ),
        "workflow_defaults_hash": (
            before.get("workflow_defaults_hash"),
            after.get("workflow_defaults_hash"),
        ),
        "validation_tolerance": (
            before.get("validation_tolerance"),
            after.get("validation_tolerance"),
        ),
        "batch_size": (before.get("batch_size"), after.get("batch_size")),
        "num_workers": (before.get("num_workers"), after.get("num_workers")),
        "device": (before.get("device"), after.get("device")),
        "hardware_fingerprint": (
            before.get("hardware_fingerprint"),
            after.get("hardware_fingerprint"),
        ),
        "metric_code_version": (
            before.get("metric_code_version"),
            after.get("metric_code_version"),
        ),
    }
    differences = [
        f"{name}: before={values[0]!r}, after={values[1]!r}"
        for name, values in checks.items()
        if values[0] != values[1]
    ]
    strict = {"split_manifest_hash", "test_document_ids", "schema_version"}
    strict_differences = [
        difference for difference in differences if difference.split(":", maxsplit=1)[0] in strict
    ]
    if strict_differences and not allow_incomparable:
        raise ValueError(
            "comparison refused because locked test protocol differs: "
            + "; ".join(strict_differences)
            + ". Use --allow-incomparable-runs only for an explicitly unfair diagnostic."
        )
    return not differences, differences


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _compare_named_values(
    before: dict[str, float],
    after: dict[str, float],
) -> tuple[dict[str, float], dict[str, float | None]]:
    names = sorted(set(before) & set(after))
    return (
        {name: absolute_change(before[name], after[name]) for name in names},
        {name: relative_change_percent(before[name], after[name]) for name in names},
    )


def compare_runs(
    before_dir: Path,
    after_dir: Path,
    output_dir: Path,
    allow_incomparable_runs: bool = False,
) -> dict[str, Any]:
    before_manifest = _load_object(before_dir / "manifest.json")
    after_manifest = _load_object(after_dir / "manifest.json")
    before_metrics_payload = _load_object(before_dir / "metrics.json")
    after_metrics_payload = _load_object(after_dir / "metrics.json")
    before_timing_payload = _load_object(before_dir / "timing.json")
    after_timing_payload = _load_object(after_dir / "timing.json")
    fair, fairness_differences = _fairness_checks(
        before_manifest, after_manifest, allow_incomparable_runs
    )
    metrics_before = _numeric_values(before_metrics_payload)
    metrics_after = _numeric_values(after_metrics_payload)
    metric_absolute, metric_relative = _compare_named_values(metrics_before, metrics_after)
    runtime_before = _timing_values(before_timing_payload)
    runtime_after = _timing_values(after_timing_payload)
    runtime_absolute, runtime_relative = _compare_named_values(runtime_before, runtime_after)
    per_document_before = before_metrics_payload.get("per_document", {})
    per_document_after = after_metrics_payload.get("per_document", {})
    primary_metric = str(
        after_metrics_payload.get(
            "primary_metric", before_metrics_payload.get("primary_metric", "")
        )
    )
    document_rows: list[dict[str, Any]] = []
    document_counts = {"improved": 0, "regressed": 0, "unchanged": 0}
    if isinstance(per_document_before, dict) and isinstance(per_document_after, dict):
        for document_id in sorted(set(per_document_before) & set(per_document_after)):
            before_entry = per_document_before[document_id]
            after_entry = per_document_after[document_id]
            if not isinstance(before_entry, dict) or not isinstance(after_entry, dict):
                continue
            before_value = before_entry.get(primary_metric)
            after_value = after_entry.get(primary_metric)
            if not isinstance(before_value, (int, float)) or not isinstance(
                after_value, (int, float)
            ):
                continue
            classification = classify_change(
                primary_metric, float(before_value), float(after_value)
            )
            document_counts[classification] += 1
            document_rows.append(
                {
                    "document_id": document_id,
                    "metric": primary_metric,
                    "before": before_value,
                    "after": after_value,
                    "absolute_change": absolute_change(float(before_value), float(after_value)),
                    "relative_change_percent": relative_change_percent(
                        float(before_value), float(after_value)
                    ),
                    "classification": classification,
                }
            )
    per_field_before = before_metrics_payload.get("per_field", {})
    per_field_after = after_metrics_payload.get("per_field", {})
    field_rows: list[dict[str, Any]] = []
    if isinstance(per_field_before, dict) and isinstance(per_field_after, dict):
        for field in sorted(set(per_field_before) & set(per_field_after)):
            before_value = per_field_before[field]
            after_value = per_field_after[field]
            if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
                field_rows.append(
                    {
                        "field": field,
                        "before": before_value,
                        "after": after_value,
                        "absolute_change": absolute_change(float(before_value), float(after_value)),
                        "relative_change_percent": relative_change_percent(
                            float(before_value), float(after_value)
                        ),
                        "classification": classify_change(
                            f"{field}_accuracy",
                            float(before_value),
                            float(after_value),
                        ),
                    }
                )
    before_checkpoint = before_manifest.get("checkpoint", {})
    after_checkpoint = after_manifest.get("checkpoint", {})
    comparison: dict[str, Any] = {
        "experiment_id": after_manifest.get("experiment_id"),
        "stage": after_manifest.get("stage"),
        "pipeline": after_manifest.get(
            "pipeline", {"detector": None, "recognizer": None, "layout": None}
        ),
        "baseline_mode": before_manifest.get("baseline_mode"),
        "finetuned_mode": after_manifest.get("finetuned_mode"),
        "pretrained_checkpoint": {
            "identifier": before_checkpoint.get("identifier"),
            "revision": before_checkpoint.get("revision"),
            "sha256": before_checkpoint.get("sha256"),
        },
        "finetuned_checkpoint": {
            "path": after_checkpoint.get("path"),
            "sha256": after_checkpoint.get("sha256"),
            "best_epoch": after_checkpoint.get("best_epoch"),
        },
        "split_manifest_hash": after_manifest.get("split_manifest_hash"),
        "same_test_split": (
            before_manifest.get("split_manifest_hash") == after_manifest.get("split_manifest_hash")
            and sorted(before_manifest.get("test_document_ids", []))
            == sorted(after_manifest.get("test_document_ids", []))
        ),
        "test_document_count": len(after_manifest.get("test_document_ids", [])),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "absolute_change": metric_absolute,
        "relative_change_percent": metric_relative,
        "runtime_before": runtime_before,
        "runtime_after": runtime_after,
        "runtime_absolute_change": runtime_absolute,
        "runtime_relative_change_percent": runtime_relative,
        "runtime_unavailable_before": before_timing_payload.get("unavailable_reasons", {}),
        "runtime_unavailable_after": after_timing_payload.get("unavailable_reasons", {}),
        "improved_document_count": document_counts["improved"],
        "regressed_document_count": document_counts["regressed"],
        "unchanged_document_count": document_counts["unchanged"],
        "failed_before_count": int(before_timing_payload.get("failed_document_count", 0)),
        "failed_after_count": int(after_timing_payload.get("failed_document_count", 0)),
        "fair_comparison": fair,
        "fairness_differences": fairness_differences,
        "comparison_contract_hash": canonical_json_hash(
            {
                "before": before_manifest,
                "after": after_manifest,
                "metric_names": sorted(set(metrics_before) | set(metrics_after)),
            }
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    comparison_rows = [
        {
            "metric": name,
            "before": metrics_before.get(name),
            "after": metrics_after.get(name),
            "absolute_change": metric_absolute.get(name),
            "relative_change_percent": metric_relative.get(name),
            "lower_is_better": lower_is_better(
                name, _metric_entries(after_metrics_payload).get(name)
            ),
            "classification": (
                classify_change(name, metrics_before[name], metrics_after[name])
                if name in metrics_before and name in metrics_after
                else "N/A"
            ),
        }
        for name in sorted(set(metrics_before) | set(metrics_after))
    ]
    _write_rows(
        output_dir / "comparison.csv",
        comparison_rows,
        [
            "metric",
            "before",
            "after",
            "absolute_change",
            "relative_change_percent",
            "lower_is_better",
            "classification",
        ],
    )
    _write_rows(
        output_dir / "per_document.csv",
        document_rows,
        [
            "document_id",
            "metric",
            "before",
            "after",
            "absolute_change",
            "relative_change_percent",
            "classification",
        ],
    )
    _write_rows(
        output_dir / "per_field.csv",
        field_rows,
        [
            "field",
            "before",
            "after",
            "absolute_change",
            "relative_change_percent",
            "classification",
        ],
    )
    timing_rows = [
        {
            "metric": name,
            "before": runtime_before.get(name),
            "after": runtime_after.get(name),
            "absolute_change": runtime_absolute.get(name),
            "relative_change_percent": runtime_relative.get(name),
            "classification": (
                classify_change(name, runtime_before[name], runtime_after[name])
                if name in runtime_before and name in runtime_after
                else "N/A"
            ),
        }
        for name in sorted(set(runtime_before) | set(runtime_after))
    ]
    _write_rows(
        output_dir / "timing_comparison.csv",
        timing_rows,
        [
            "metric",
            "before",
            "after",
            "absolute_change",
            "relative_change_percent",
            "classification",
        ],
    )
    improved_metrics = [
        str(row["metric"]) for row in comparison_rows if row["classification"] == "improved"
    ]
    regressed_metrics = [
        str(row["metric"]) for row in comparison_rows if row["classification"] == "regressed"
    ]
    quality_available = bool(metrics_before) and bool(metrics_after)
    runtime_improved = [
        str(row["metric"]) for row in timing_rows if row["classification"] == "improved"
    ]
    runtime_regressed = [
        str(row["metric"]) for row in timing_rows if row["classification"] == "regressed"
    ]
    if not quality_available:
        quality_conclusion = (
            "insufficient valid test ground truth; no fine-tuning quality conclusion"
        )
    elif improved_metrics and not regressed_metrics:
        quality_conclusion = "fine-tuning improved the measured quality metrics"
    elif improved_metrics:
        quality_conclusion = "fine-tuning produced mixed quality changes"
    else:
        quality_conclusion = "fine-tuning did not improve the measured quality metrics"
    summary = [
        "# Pretrained vs fine-tuned",
        "",
        f"- Fair comparison: {'yes' if fair else 'no'}",
        f"- Same locked test split: {'yes' if comparison['same_test_split'] else 'no'}",
        (
            "- Có đủ test ground truth để so sánh chất lượng."
            if quality_available
            else "- Không đủ test ground truth hợp lệ; không kết luận fine-tuning tốt hơn."
        ),
        f"- Metrics improved: {', '.join(improved_metrics) if improved_metrics else 'none'}",
        f"- Metrics regressed: {', '.join(regressed_metrics) if regressed_metrics else 'none'}",
        f"- Quality conclusion: {quality_conclusion}",
        (
            "- Runtime/throughput improved: "
            + (", ".join(runtime_improved) if runtime_improved else "none")
        ),
        (
            "- Runtime/throughput/memory regressed: "
            + (", ".join(runtime_regressed) if runtime_regressed else "none")
        ),
        (f"- Unavailable resources before: {comparison['runtime_unavailable_before']}"),
        (f"- Unavailable resources after: {comparison['runtime_unavailable_after']}"),
        f"- Documents improved: {document_counts['improved']}",
        f"- Documents regressed: {document_counts['regressed']}",
        f"- Documents unchanged: {document_counts['unchanged']}",
        (
            "- Failed before/after: "
            f"{comparison['failed_before_count']}/{comparison['failed_after_count']}"
        ),
    ]
    if fairness_differences:
        summary.append("- Fairness differences: " + "; ".join(fairness_differences))
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return comparison

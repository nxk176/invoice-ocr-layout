from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from invoice_ocr.experiments.comparison import (
    absolute_change,
    classify_change,
    compare_runs,
    relative_change_percent,
)


def _write_run(
    root: Path,
    split_hash: str,
    cer: float,
    f1: float,
    document_values: dict[str, float],
) -> None:
    root.mkdir(parents=True)
    manifest = {
        "experiment_id": "synthetic",
        "stage": "recognizer",
        "pipeline": {"detector": None, "recognizer": "vietocr", "layout": None},
        "baseline_mode": "pretrained",
        "finetuned_mode": "full_finetune",
        "checkpoint": {"identifier": "synthetic", "revision": "abc123", "sha256": None},
        "split_manifest_hash": split_hash,
        "test_document_ids": sorted(document_values),
        "schema_version": "schema-v1",
        "preprocessing_config_hash": "pre-v1",
        "postprocessing_config_hash": "post-v1",
        "workflow_defaults_hash": "workflow-v1",
        "validation_tolerance": "0.01",
        "batch_size": 1,
        "num_workers": 0,
        "device": "cpu",
        "hardware_fingerprint": "cpu-a",
        "metric_code_version": "metrics-v1",
    }
    metrics = {
        "metrics": {
            "recognition_cer": {
                "value": cer,
                "lower_is_better": True,
                "numerator": cer * 100,
                "denominator": 100,
                "evaluated_sample_count": len(document_values),
                "skipped_sample_count": 0,
                "na_reason": None,
            },
            "entity_f1": {
                "value": f1,
                "lower_is_better": False,
                "numerator": None,
                "denominator": None,
                "evaluated_sample_count": len(document_values),
                "skipped_sample_count": 0,
                "na_reason": None,
            },
        },
        "primary_metric": "recognition_cer",
        "per_document": {
            document_id: {"recognition_cer": value}
            for document_id, value in document_values.items()
        },
        "per_field": {"invoice_number": f1},
    }
    timing = {
        "total_wall_time_seconds": 10.0,
        "throughput_documents_per_second": 0.2,
        "peak_cpu_ram_mb": 100.0,
        "failed_document_count": 0,
    }
    for name, payload in (
        ("manifest.json", manifest),
        ("metrics.json", metrics),
        ("timing.json", timing),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


def test_delta_relative_and_metric_direction() -> None:
    assert absolute_change(0.5, 0.4) == pytest.approx(-0.1)
    assert relative_change_percent(0.5, 0.4) == pytest.approx(-20.0)
    assert relative_change_percent(0.0, 1.0) is None
    assert classify_change("recognition_cer", 0.5, 0.4) == "improved"
    assert classify_change("entity_f1", 0.5, 0.4) == "regressed"


def test_comparison_counts_documents_and_lower_is_better(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_run(before, "same-split", 0.5, 0.6, {"a": 0.5, "b": 0.2, "c": 0.1})
    _write_run(after, "same-split", 0.4, 0.7, {"a": 0.4, "b": 0.3, "c": 0.1})
    result = compare_runs(before, after, tmp_path / "comparison")
    assert result["absolute_change"]["recognition_cer"] == pytest.approx(-0.1)
    assert result["relative_change_percent"]["recognition_cer"] == pytest.approx(-20)
    assert result["improved_document_count"] == 1
    assert result["regressed_document_count"] == 1
    assert result["unchanged_document_count"] == 1
    with (tmp_path / "comparison" / "comparison.csv").open(encoding="utf-8") as stream:
        rows = {row["metric"]: row for row in csv.DictReader(stream)}
    assert rows["recognition_cer"]["classification"] == "improved"
    assert rows["recognition_cer"]["lower_is_better"] == "True"


def test_comparison_refuses_different_locked_split_by_default(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_run(before, "split-a", 0.5, 0.6, {"a": 0.5})
    _write_run(after, "split-b", 0.4, 0.7, {"a": 0.4})
    with pytest.raises(ValueError, match="comparison refused"):
        compare_runs(before, after, tmp_path / "refused")
    result = compare_runs(
        before,
        after,
        tmp_path / "allowed",
        allow_incomparable_runs=True,
    )
    assert result["fair_comparison"] is False
    assert result["same_test_split"] is False

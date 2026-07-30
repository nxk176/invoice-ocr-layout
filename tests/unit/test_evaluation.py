from __future__ import annotations

import json
from pathlib import Path

from invoice_ocr.contracts import BoundingBox
from invoice_ocr.evaluation.benchmark import benchmark_rows, write_benchmark_reports
from invoice_ocr.evaluation.detection import box_iou, detection_metrics
from invoice_ocr.evaluation.final_json import final_json_metrics
from invoice_ocr.evaluation.layout import set_metrics


def test_detection_metrics_match_boxes_once() -> None:
    exact = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    distant = BoundingBox(x_min=20, y_min=20, x_max=30, y_max=30)
    assert box_iou(exact, exact) == 1
    metric = detection_metrics([exact, distant], [exact])
    assert metric.precision == 0.5
    assert metric.recall == 1
    assert metric.f1 == 2 / 3


def test_layout_set_metrics() -> None:
    metric = set_metrics({("A",), ("B",)}, {("A",), ("C",)})
    assert metric.precision == 0.5
    assert metric.recall == 0.5
    assert metric.f1 == 0.5


def test_final_json_metrics_normalize_only_string_whitespace() -> None:
    expected = {"field": "Synthetic Value", "number": "0001"}
    predicted = {"field": " synthetic   value ", "number": "0001"}
    metric = final_json_metrics(predicted, expected)
    assert metric["field_exact_match"] == 0.5
    assert metric["normalized_field_accuracy"] == 1
    assert metric["document_exact_match"] == 0


def test_benchmark_writes_all_required_reports_with_na_reasons(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    gt = tmp_path / "GT"
    gt.mkdir()
    rows = benchmark_rows(True)
    write_benchmark_reports(output, rows, gt)
    assert len(rows) == 12
    for filename in ("metrics.json", "metrics.csv", "comparison.csv", "summary.md"):
        assert (output / filename).is_file()
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert payload["pipeline_count"] == 12
    assert payload["stage_metric_availability"]["detection"]["status"] == "N/A"
    assert payload["stage_metric_availability"]["detection"]["reason"]

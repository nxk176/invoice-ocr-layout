"""Aggregate available GT levels for one completed pipeline run."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from invoice_ocr.contracts import BoundingBox, DetectionRegion, LabeledEntity, RecognizedRegion
from invoice_ocr.evaluation.detection import detection_metrics
from invoice_ocr.evaluation.final_json import final_json_metrics
from invoice_ocr.evaluation.layout import set_metrics
from invoice_ocr.evaluation.recognition import recognition_metrics
from invoice_ocr.io.jsonl import read_jsonl


def _load_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: sum(row[key] for row in rows if key in row) / sum(key in row for row in rows)
        for key in keys
    }


def _bbox(raw: dict[str, Any]) -> BoundingBox:
    box = raw.get("bbox")
    if isinstance(box, dict):
        return BoundingBox.model_validate(box)
    polygon = raw.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("detection GT region needs bbox or polygon")
    x_values = [float(point["x"] if isinstance(point, dict) else point[0]) for point in polygon]
    y_values = [float(point["y"] if isinstance(point, dict) else point[1]) for point in polygon]
    return BoundingBox(
        x_min=min(x_values),
        y_min=min(y_values),
        x_max=max(x_values),
        y_max=max(y_values),
    )


def _evaluate_detection(run_dir: Path, gt_root: Path) -> dict[str, Any]:
    paths = sorted((gt_root / "detection").glob("*.json"))
    if not paths:
        return {"detection_status": "N/A", "detection_reason": "GT/detection missing"}
    predicted = list(
        read_jsonl(
            run_dir.parent.parent / "work" / run_dir.name / "detections.jsonl", DetectionRegion
        )
    )
    by_page: dict[tuple[str, int], list[BoundingBox]] = defaultdict(list)
    for region in predicted:
        by_page[(region.document_id, region.page_index)].append(region.bbox)
    values = []
    for path in paths:
        payload = _load_object(path)
        document_id = str(payload.get("document_id", path.stem))
        for page in payload["pages"]:
            page_index = int(page.get("page_index", 0))
            expected = [_bbox(region) for region in page["regions"]]
            metric = detection_metrics(by_page[(document_id, page_index)], expected)
            values.append(
                {
                    "detection_iou": metric.mean_iou,
                    "detection_precision": metric.precision,
                    "detection_recall": metric.recall,
                    "detection_f1": metric.f1,
                }
            )
    return {"detection_status": "available", **_average(values)}


def _evaluate_recognition(run_dir: Path, gt_root: Path) -> dict[str, Any]:
    paths = sorted((gt_root / "recognition").glob("*.json"))
    if not paths:
        return {"recognition_status": "N/A", "recognition_reason": "GT/recognition missing"}
    predictions = list(
        read_jsonl(
            run_dir.parent.parent / "work" / run_dir.name / "recognitions.jsonl",
            RecognizedRegion,
        )
    )
    by_document: dict[str, list[RecognizedRegion]] = defaultdict(list)
    for region in predictions:
        by_document[region.document_id].append(region)
    rows = []
    for path in paths:
        payload = _load_object(path)
        document_id = str(payload.get("document_id", path.stem))
        expected_regions = payload["regions"]
        indexed = {region.region_id: region.text for region in by_document[document_id]}
        predicted_text = []
        expected_text = []
        for index, region in enumerate(expected_regions):
            region_id = str(region.get("region_id", ""))
            predicted_text.append(
                indexed.get(
                    region_id,
                    by_document[document_id][index].text
                    if index < len(by_document[document_id])
                    else "",
                )
            )
            expected_text.append(str(region["text"]))
        metric = recognition_metrics(predicted_text, expected_text)
        rows.append(
            {
                "recognition_cer": metric.cer,
                "recognition_wer": metric.wer,
                "recognition_exact_match": metric.exact_match,
                **(
                    {"recognition_numeric_accuracy": metric.numeric_field_accuracy}
                    if metric.numeric_field_accuracy is not None
                    else {}
                ),
            }
        )
    return {"recognition_status": "available", **_average(rows)}


def _evaluate_layout(run_dir: Path, gt_root: Path) -> dict[str, Any]:
    paths = sorted((gt_root / "layout").glob("*.json"))
    if not paths:
        return {"layout_status": "N/A", "layout_reason": "GT/layout missing"}
    entities = list(
        read_jsonl(
            run_dir.parent.parent / "work" / run_dir.name / "entities.jsonl",
            LabeledEntity,
        )
    )
    predicted = {
        (entity.document_id, entity.page_index, entity.label, " ".join(entity.text.split()))
        for entity in entities
    }
    expected: set[tuple[object, ...]] = set()
    has_relations = False
    for path in paths:
        payload = _load_object(path)
        document_id = str(payload.get("document_id", path.stem))
        has_relations = has_relations or bool(payload.get("relations"))
        for page in payload["pages"]:
            page_index = int(page.get("page_index", 0))
            for token, label in zip(page["tokens"], page["labels"], strict=True):
                if label != "O":
                    expected.add((document_id, page_index, str(label), str(token).strip()))
    metric = set_metrics(predicted, expected)
    result: dict[str, Any] = {
        "layout_status": "available",
        "entity_precision": metric.precision,
        "entity_recall": metric.recall,
        "entity_f1": metric.f1,
        "field_exact_match": metric.f1,
    }
    if has_relations:
        result["relation_status"] = "N/A"
        result["relation_reason"] = "pipeline adapter did not persist predicted relations"
    else:
        result["relation_status"] = "N/A"
        result["relation_reason"] = "relation annotations are missing"
    return result


def _evaluate_final(run_dir: Path, gt_root: Path) -> dict[str, Any]:
    gt_final = gt_root / "final"
    paths = sorted(gt_final.rglob("*.json")) if gt_final.is_dir() else []
    if not paths:
        return {"final_status": "N/A", "final_reason": "GT/final missing"}
    rows = []
    validation_checks = 0
    validation_successes = 0
    missing = 0
    for path in paths:
        prediction = run_dir / "predictions" / path.relative_to(gt_final)
        if not prediction.is_file():
            missing += 1
            continue
        predicted = _load_object(prediction)
        expected = _load_object(path)
        rows.append(final_json_metrics(predicted, expected))
        for invoice in predicted.get("invoices", []):
            validation = invoice.get("validation", {})
            for field in (
                "subtotal_plus_vat_equals_grand_total",
                "sum_of_items_equals_subtotal",
            ):
                if validation.get(field) is not None:
                    validation_checks += 1
                    validation_successes += validation.get(field) is True
    result: dict[str, Any] = {
        "final_status": "available",
        "final_missing_predictions": missing,
        **_average(rows),
    }
    result["validation_success_rate"] = (
        validation_successes / validation_checks if validation_checks else None
    )
    return result


def evaluate_run(run_dir: Path, gt_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for evaluator in (
        _evaluate_detection,
        _evaluate_recognition,
        _evaluate_layout,
        _evaluate_final,
    ):
        try:
            result.update(evaluator(run_dir, gt_root))
        except (OSError, ValueError, KeyError) as exc:
            stage = evaluator.__name__.removeprefix("_evaluate_")
            result[f"{stage}_status"] = "N/A"
            result[f"{stage}_reason"] = f"evaluation failed: {exc}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        runtime = _load_object(metrics_path)
        result["processing_seconds"] = runtime.get("processing_seconds")
        result["seconds_per_document"] = runtime.get("seconds_per_document")
        result["failed_document_count"] = runtime.get("failed_document_count")
    return result

"""Standalone stage commands operating on stable JSONL contracts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from invoice_ocr.adapters.detectors import DETECTORS
from invoice_ocr.adapters.layout import LAYOUT_ADAPTERS
from invoice_ocr.adapters.recognizers import RECOGNIZERS
from invoice_ocr.contracts import (
    DetectionRegion,
    DocumentPage,
    InvoiceBatch,
    LabeledEntity,
    RecognizedRegion,
)
from invoice_ocr.exceptions import ConfigurationError
from invoice_ocr.io.jsonl import read_jsonl, write_jsonl
from invoice_ocr.io.paths import discover_documents
from invoice_ocr.io.pdf_render import render_document
from invoice_ocr.pipeline import (
    entities_to_table_cells,
    resolve_device,
    validate_canonical_payload,
    write_json,
)
from invoice_ocr.postprocessing.fields import entities_to_invoice, load_workflow_defaults
from invoice_ocr.postprocessing.validation import validate_invoice
from invoice_ocr.reconstruction.medicine_rows import reconstruct_medicine_item, reconstruct_rows


def run_detect_stage(
    detector_name: str,
    input_path: Path,
    output_dir: Path,
    model_root: Path,
    device: str,
) -> Path:
    detector = DETECTORS[detector_name](model_root, resolve_device(device))
    detector.prepare()
    pages_path = output_dir / "pages.jsonl"
    detections_path = output_dir / "detections.jsonl"
    for document in discover_documents(input_path):
        pages = render_document(document, output_dir / "rendered" / document.document_id)
        write_jsonl(pages_path, pages, append=True)
        for page in pages:
            write_jsonl(detections_path, detector.detect(page), append=True)
    return detections_path


def run_recognize_stage(
    recognizer_name: str,
    input_dir: Path,
    output_dir: Path,
    model_root: Path,
    device: str,
) -> Path:
    pages = list(read_jsonl(input_dir / "pages.jsonl", DocumentPage))
    detections = list(read_jsonl(input_dir / "detections.jsonl", DetectionRegion))
    if not pages:
        raise ConfigurationError(f"no page records found at {input_dir / 'pages.jsonl'}")
    by_page: dict[tuple[str, int], list[DetectionRegion]] = defaultdict(list)
    for region in detections:
        by_page[(region.document_id, region.page_index)].append(region)
    recognizer = RECOGNIZERS[recognizer_name](model_root, resolve_device(device))
    recognizer.prepare()
    target = output_dir / "recognitions.jsonl"
    for page in pages:
        regions = by_page[(page.document_id, page.page_index)]
        write_jsonl(target, recognizer.recognize(page, regions), append=True)
    return target


def run_extract_stage(
    layout_name: str,
    input_dir: Path,
    output_dir: Path,
    model_root: Path,
    device: str,
) -> Path:
    pages = list(read_jsonl(input_dir / "pages.jsonl", DocumentPage))
    recognized = list(read_jsonl(input_dir / "recognitions.jsonl", RecognizedRegion))
    if not pages:
        raise ConfigurationError(f"no page records found at {input_dir / 'pages.jsonl'}")
    by_page: dict[tuple[str, int], list[RecognizedRegion]] = defaultdict(list)
    for region in recognized:
        by_page[(region.document_id, region.page_index)].append(region)
    layout = LAYOUT_ADAPTERS[layout_name](model_root, resolve_device(device))
    layout.prepare()
    target = output_dir / "entities.jsonl"
    for page in pages:
        entities, _relations = layout.extract(page, by_page[(page.document_id, page.page_index)])
        write_jsonl(target, entities, append=True)
    return target


def run_postprocess_stage(
    input_dir: Path,
    output_dir: Path,
    workflow_defaults: Path | None,
) -> Path:
    entities = list(read_jsonl(input_dir / "entities.jsonl", LabeledEntity))
    if not entities:
        raise ConfigurationError(f"no entity records found at {input_dir / 'entities.jsonl'}")
    defaults = load_workflow_defaults(workflow_defaults)
    by_document_page: dict[tuple[str, int], list[LabeledEntity]] = defaultdict(list)
    source_by_document: dict[str, str] = {}
    for entity in entities:
        by_document_page[(entity.document_id, entity.page_index)].append(entity)
        source_by_document[entity.document_id] = entity.source_path
    prediction_root = output_dir / "predictions"
    for document_id in sorted(source_by_document):
        invoices = []
        for (candidate_id, page_index), page_entities in sorted(by_document_page.items()):
            if candidate_id != document_id:
                continue
            invoice = entities_to_invoice(page_index + 1, page_entities, defaults)
            rows = reconstruct_rows(entities_to_table_cells(page_entities))
            invoice.items = [
                reconstruct_medicine_item(row, line_number=index)
                for index, row in enumerate(rows, start=1)
            ]
            validate_invoice(invoice)
            invoices.append(invoice)
        batch = InvoiceBatch(invoice_count=len(invoices), invoices=invoices)
        payload = batch.model_dump(mode="python", exclude_none=False)
        validate_canonical_payload(payload)
        write_json(prediction_root / f"{document_id}.json", payload)
    return prediction_root


def evaluate_prediction_directory(
    prediction_dir: Path,
    gt_root: Path,
    output_path: Path,
) -> dict[str, object]:
    from invoice_ocr.evaluation.final_json import final_json_metrics

    gt_final = gt_root / "final"
    expected_paths = sorted(gt_final.rglob("*.json")) if gt_final.is_dir() else []
    if not expected_paths:
        metrics: dict[str, object] = {
            "final": {"status": "N/A", "reason": "GT/final annotations are missing"}
        }
        write_json(output_path, metrics)
        return metrics
    document_metrics = []
    missing = 0
    for expected_path in expected_paths:
        relative = expected_path.relative_to(gt_final)
        predicted_path = prediction_dir / relative
        if not predicted_path.is_file():
            missing += 1
            continue
        predicted = json.loads(predicted_path.read_text(encoding="utf-8"))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        document_metrics.append(final_json_metrics(predicted, expected))
    keys = sorted({key for metrics in document_metrics for key in metrics})
    aggregate = (
        {
            key: sum(metrics[key] for metrics in document_metrics) / len(document_metrics)
            for key in keys
        }
        if document_metrics
        else {}
    )
    result = {
        "final": {
            "status": "available",
            "evaluated_documents": len(document_metrics),
            "missing_predictions": missing,
            **aggregate,
        }
    }
    write_json(output_path, result)
    return result

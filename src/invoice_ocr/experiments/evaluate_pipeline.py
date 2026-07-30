"""Locked-test end-to-end canonical pipeline evaluation."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from invoice_ocr.adapters.detectors import DETECTORS
from invoice_ocr.adapters.layout import LAYOUT_ADAPTERS
from invoice_ocr.adapters.recognizers import RECOGNIZERS
from invoice_ocr.contracts import DocumentError, InvoiceBatch
from invoice_ocr.evaluation.final_json import (
    final_json_metrics,
    flatten_json,
    normalize_scalar,
)
from invoice_ocr.experiments.contracts import AggregateMetric
from invoice_ocr.experiments.hashing import (
    canonical_json_hash,
    directory_manifest_hash,
    file_hash_or_missing,
    reproducibility_metadata,
)
from invoice_ocr.experiments.runtime import EvaluationTimer
from invoice_ocr.experiments.split import assert_locked_dataset_matches, load_locked_split
from invoice_ocr.io.paths import discover_documents, prediction_relative_path
from invoice_ocr.io.pdf_render import render_document
from invoice_ocr.model_manifest import load_adapter_manifest
from invoice_ocr.pipeline import (
    PipelineSelection,
    entities_to_table_cells,
    resolve_device,
    validate_canonical_payload,
)
from invoice_ocr.postprocessing.fields import (
    entities_to_invoice,
    load_workflow_defaults,
)
from invoice_ocr.postprocessing.validation import validate_invoice
from invoice_ocr.reconstruction.medicine_rows import (
    reconstruct_medicine_item,
    reconstruct_rows,
)


@dataclass
class PipelineEvaluationRequest:
    pipeline: PipelineSelection
    checkpoints: dict[str, Path | None]
    run_kind: str
    data_root: Path
    gt_root: Path
    split_manifest: Path
    output_dir: Path
    model_root: Path = Path("models")
    work_root: Path = Path("work")
    workflow_defaults: Path | None = None
    device: str = "auto"
    batch_size: int = 1
    num_workers: int = 0
    warmup_iterations: int = 0
    seed: int = 42
    resume: bool = False
    force: bool = False
    validation_tolerance: str = "0.01"
    baseline_mode: str | None = None
    finetuned_mode: str | None = None


def _load_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _write_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _metric(
    value: float | None,
    numerator: float | int | None,
    denominator: float | int | None,
    evaluated: int,
    skipped: int,
    lower: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    return AggregateMetric(
        value=value,
        numerator=numerator,
        denominator=denominator,
        evaluated_sample_count=evaluated,
        skipped_sample_count=skipped,
        lower_is_better=lower,
        na_reason=reason,
    ).model_dump(mode="json")


def _checkpoint_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.is_dir():
        return directory_manifest_hash(path)
    return file_hash_or_missing(path)


def _resume_complete(output_dir: Path) -> bool:
    required = ("manifest.json", "metrics.json", "timing.json")
    if not all((output_dir / name).is_file() for name in required):
        return False
    return _load_object(output_dir / "manifest.json").get("status") == "success"


def _final_gt_path(gt_root: Path, relative_path: str) -> Path:
    return gt_root / "final" / Path(relative_path).with_suffix(".json")


def _final_na_metrics(reason: str, skipped: int) -> dict[str, Any]:
    return {
        name: _metric(None, None, None, 0, skipped, lower, reason)
        for name, lower in (
            ("final_field_exact_match", False),
            ("final_normalized_field_accuracy", False),
            ("final_medicine_row_matching", False),
            ("final_document_exact_match", False),
        )
    }


def _aggregate_final(
    predictions: dict[str, dict[str, Any]],
    expected_by_document: dict[str, dict[str, Any]],
    skipped_count: int,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, float]]:
    if not expected_by_document:
        reason = "canonical final test ground truth is missing under GT/final"
        return _final_na_metrics(reason, skipped_count), {}, {}
    rows: list[dict[str, float]] = []
    per_document: dict[str, dict[str, float]] = {}
    field_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    missing_predictions = 0
    for document_id, expected in expected_by_document.items():
        predicted = predictions.get(document_id)
        if predicted is None:
            missing_predictions += 1
            continue
        document_metrics = final_json_metrics(predicted, expected)
        renamed = {f"final_{name}": value for name, value in document_metrics.items()}
        rows.append(renamed)
        per_document[document_id] = renamed
        predicted_fields = flatten_json(predicted)
        expected_fields = flatten_json(expected)
        for field, expected_value in expected_fields.items():
            field_counts[field][1] += 1
            field_counts[field][0] += normalize_scalar(
                predicted_fields.get(field)
            ) == normalize_scalar(expected_value)
    evaluated = len(rows)
    metrics: dict[str, Any] = {}
    for name in (
        "final_field_exact_match",
        "final_normalized_field_accuracy",
        "final_medicine_row_matching",
        "final_document_exact_match",
    ):
        values = [row[name] for row in rows]
        total = sum(values)
        metrics[name] = _metric(
            total / evaluated if evaluated else None,
            total if evaluated else None,
            evaluated if evaluated else None,
            evaluated,
            skipped_count + missing_predictions,
            False,
            None if evaluated else "no prediction matched canonical final test ground truth",
        )
    per_field = {
        field: correct / total for field, (correct, total) in field_counts.items() if total
    }
    return metrics, per_document, per_field


def _validation_metrics(
    predictions: dict[str, dict[str, Any]],
    processed_count: int,
    skipped_count: int,
) -> dict[str, Any]:
    validation_checks = validation_successes = unresolved = 0
    schema_success = 0
    for prediction in predictions.values():
        try:
            validate_canonical_payload(prediction)
            schema_success += 1
        except Exception:
            pass
        for invoice in prediction.get("invoices", []):
            validation = invoice.get("validation", {})
            unresolved += len(validation.get("unresolved_required_fields", []))
            for field in (
                "subtotal_plus_vat_equals_grand_total",
                "sum_of_items_equals_subtotal",
            ):
                if validation.get(field) is not None:
                    validation_checks += 1
                    validation_successes += validation.get(field) is True
    return {
        "validation_success_rate": _metric(
            (validation_successes / validation_checks if validation_checks else None),
            validation_successes if validation_checks else None,
            validation_checks if validation_checks else None,
            validation_checks,
            skipped_count,
            False,
            (
                None
                if validation_checks
                else "predictions contain no arithmetic validation checks with complete values"
            ),
        ),
        "unresolved_required_field_count": _metric(
            float(unresolved) if processed_count else None,
            unresolved if processed_count else None,
            processed_count if processed_count else None,
            processed_count,
            skipped_count,
            True,
            None if processed_count else "no successful canonical predictions",
        ),
        "schema_validation_success_rate": _metric(
            schema_success / processed_count if processed_count else None,
            schema_success if processed_count else None,
            processed_count if processed_count else None,
            processed_count,
            skipped_count,
            False,
            None if processed_count else "no successful canonical predictions",
        ),
    }


def evaluate_pipeline(request: PipelineEvaluationRequest) -> Path:
    """Run the same locked test IDs through A -> B -> C and canonical post-processing."""
    started_at = datetime.now(timezone.utc)
    if request.output_dir.exists() and request.resume and _resume_complete(request.output_dir):
        return request.output_dir
    if (
        request.output_dir.exists()
        and not request.force
        and not request.resume
        and any(request.output_dir.iterdir())
    ):
        raise ValueError(
            f"pipeline evaluation output exists: {request.output_dir}; use --resume or --force"
        )
    split = load_locked_split(request.split_manifest)
    assert_locked_dataset_matches(split, request.data_root, request.gt_root)
    test_ids = set(split.test_document_ids)
    if not test_ids:
        raise ValueError("locked test split contains no document IDs")
    documents = [
        document
        for document in discover_documents(request.data_root)
        if document.document_id in test_ids
    ]
    if {document.document_id for document in documents} != test_ids:
        raise ValueError("one or more locked test documents are absent from data")
    request.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = request.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(request.device)
    timer = EvaluationTimer(request.warmup_iterations)
    timer.start()
    with timer.stage("model_load"):
        detector = DETECTORS[request.pipeline.detector](
            request.model_root,
            device,
            request.checkpoints.get("detector"),
        )
        recognizer = RECOGNIZERS[request.pipeline.recognizer](
            request.model_root,
            device,
            request.checkpoints.get("recognizer"),
        )
        layout = LAYOUT_ADAPTERS[request.pipeline.layout](
            request.model_root,
            device,
            request.checkpoints.get("layout"),
        )
        detector.prepare()
        recognizer.prepare()
        layout.prepare()
    defaults = load_workflow_defaults(request.workflow_defaults)
    tolerance = Decimal(str(request.validation_tolerance))
    required_fields = [str(value) for value in defaults.get("required_fields", [])]
    rendered: dict[str, list[Any]] = {}
    with timer.stage("preprocessing"):
        for document in documents:
            rendered[document.document_id] = render_document(
                document,
                request.work_root / request.output_dir.name / document.document_id,
            )
    first_page = next(iter(rendered.values()))[0]
    for _ in range(request.warmup_iterations):
        warm_detections = detector.detect(first_page)
        warm_recognitions = recognizer.recognize(first_page, warm_detections)
        layout.extract(first_page, warm_recognitions)
    predictions: dict[str, dict[str, Any]] = {}
    expected: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    processed_pages = 0
    failed = 0
    for document in documents:
        with timer.document():
            try:
                invoices = []
                for page in rendered[document.document_id]:
                    with timer.stage("detection"):
                        detections = detector.detect(page)
                    with timer.stage("recognition"):
                        recognitions = recognizer.recognize(page, detections)
                    with timer.stage("layout_inference"):
                        entities, _relations = layout.extract(page, recognitions)
                    with timer.stage("table_reconstruction"):
                        table_cells = entities_to_table_cells(entities)
                        rows = reconstruct_rows(table_cells)
                    with timer.stage("postprocessing"):
                        invoice = entities_to_invoice(
                            page.page_index + 1,
                            entities,
                            workflow_defaults=defaults,
                        )
                        invoice.items = [
                            reconstruct_medicine_item(row, line_number=index)
                            for index, row in enumerate(rows, start=1)
                        ]
                    with timer.stage("validation"):
                        validate_invoice(invoice, tolerance, required_fields)
                    invoices.append(invoice)
                    processed_pages += 1
                batch = InvoiceBatch(invoice_count=len(invoices), invoices=invoices)
                payload = batch.model_dump(mode="json", exclude_none=False)
                validate_canonical_payload(payload)
                predictions[document.document_id] = payload
                prediction_path = predictions_dir / prediction_relative_path(document.relative_path)
                _write_object(prediction_path, payload)
                gt_path = _final_gt_path(request.gt_root, document.relative_path)
                if gt_path.is_file():
                    expected[document.document_id] = _load_object(gt_path)
            except Exception as exc:
                failed += 1
                error = DocumentError(
                    document_id=document.document_id,
                    source_path=document.source_path,
                    stage="pipeline_experiment",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    recoverable=True,
                )
                errors.append(error.model_dump(mode="json"))
    with timer.stage("evaluation"):
        final_metrics, per_document, per_field = _aggregate_final(
            predictions,
            expected,
            len(test_ids) - len(expected),
        )
        final_metrics.update(
            _validation_metrics(
                predictions,
                len(predictions),
                len(test_ids) - len(predictions),
            )
        )
    timing = timer.finish(
        len(predictions),
        processed_pages,
        failed,
        len(test_ids) - len(predictions) - failed,
    )
    _write_object(
        request.output_dir / "metrics.json",
        {
            "metrics": final_metrics,
            "primary_metric": "final_normalized_field_accuracy",
            "per_document": per_document,
            "per_field": per_field,
        },
    )
    _write_object(request.output_dir / "timing.json", timing.model_dump(mode="json"))
    _write_jsonl(request.output_dir / "errors.jsonl", errors)
    reproducibility = reproducibility_metadata()
    schema_path = Path(__file__).resolve().parents[3] / "configs/schema/invoice.schema.json"
    preprocessing = {
        "renderer": "invoice_ocr.io.pdf_render",
        "orientation": "exif",
        "deskew": "configured_runtime",
    }
    postprocessing = {"canonical": "v1", "validation_tolerance": request.validation_tolerance}
    model_names = {
        "detector": request.pipeline.detector,
        "recognizer": request.pipeline.recognizer,
        "layout": request.pipeline.layout,
    }
    checkpoints = {
        stage: {
            "path": str(path) if path else None,
            "sha256": _checkpoint_hash(path),
            "identifier": load_adapter_manifest(stage, model_names[stage]).get(
                "checkpoint_identifier"
            ),
            "source_repository": load_adapter_manifest(stage, model_names[stage]).get(
                "official_repository"
            ),
            "source_revision": load_adapter_manifest(stage, model_names[stage]).get("revision"),
            "checkpoint_revision": load_adapter_manifest(stage, model_names[stage]).get(
                "checkpoint_revision"
            ),
        }
        for stage, path in request.checkpoints.items()
    }
    manifest = {
        "experiment_id": request.output_dir.parent.name,
        "run_kind": request.run_kind,
        "stage": None,
        "model": None,
        "pipeline": {
            "detector": request.pipeline.detector,
            "recognizer": request.pipeline.recognizer,
            "layout": request.pipeline.layout,
        },
        "baseline_mode": request.baseline_mode,
        "finetuned_mode": request.finetuned_mode,
        "checkpoint": {
            "identifier": "pipeline",
            "revision": canonical_json_hash(checkpoints),
            "sha256": canonical_json_hash(checkpoints),
            "components": checkpoints,
        },
        "split_manifest_hash": split.split_manifest_hash,
        "test_document_ids": split.test_document_ids,
        "schema_version": file_hash_or_missing(schema_path),
        "preprocessing_config_hash": canonical_json_hash(preprocessing),
        "postprocessing_config_hash": canonical_json_hash(postprocessing),
        "workflow_defaults_hash": (
            file_hash_or_missing(request.workflow_defaults)
            if request.workflow_defaults
            else canonical_json_hash({"workflow_defaults": None})
        ),
        "validation_tolerance": request.validation_tolerance,
        "batch_size": request.batch_size,
        "num_workers": request.num_workers,
        "device": device,
        "hardware_fingerprint": reproducibility["hardware_fingerprint"],
        "metric_code_version": "invoice-ocr-metrics-v2",
        "reproducibility": reproducibility,
        "random_seed": request.seed,
        "command_line": sys.argv,
        "start_time": started_at.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "data_manifest_hash": split.dataset_manifest_hash,
        "gt_manifest_hash": split.gt_manifest_hash,
        "status": "success" if predictions else "failed",
        "skip_reason": None if predictions else "all locked test documents failed",
    }
    _write_object(request.output_dir / "manifest.json", manifest)
    (request.output_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "pipeline": manifest["pipeline"],
                "checkpoints": checkpoints,
                "test_document_ids": split.test_document_ids,
                "warmup_iterations": request.warmup_iterations,
                "batch_size": request.batch_size,
                "num_workers": request.num_workers,
                "device": device,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return request.output_dir

"""End-to-end document orchestration with stable artifacts and error isolation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from invoice_ocr.adapters.detectors import DETECTORS, DetectorAdapter
from invoice_ocr.adapters.layout import LAYOUT_ADAPTERS, LayoutAdapter
from invoice_ocr.adapters.recognizers import RECOGNIZERS, RecognizerAdapter
from invoice_ocr.contracts import (
    DocumentError,
    InvoiceBatch,
    LabeledEntity,
    ModelIdentity,
    ProcessingStatus,
    RunManifest,
    TableCell,
)
from invoice_ocr.exceptions import ConfigurationError, OutputExistsError
from invoice_ocr.io.jsonl import write_jsonl
from invoice_ocr.io.paths import discover_documents, prediction_relative_path
from invoice_ocr.io.pdf_render import render_document
from invoice_ocr.postprocessing.fields import entities_to_invoice, load_workflow_defaults
from invoice_ocr.postprocessing.validation import validate_invoice
from invoice_ocr.reconstruction.medicine_rows import reconstruct_medicine_item, reconstruct_rows

DETECTOR_NAMES = tuple(DETECTORS)
RECOGNIZER_NAMES = tuple(RECOGNIZERS)
LAYOUT_NAMES = tuple(LAYOUT_ADAPTERS)


@dataclass(frozen=True)
class PipelineSelection:
    detector: str
    recognizer: str
    layout: str

    def __post_init__(self) -> None:
        if self.detector not in DETECTOR_NAMES:
            raise ConfigurationError(f"unsupported detector: {self.detector}")
        if self.recognizer not in RECOGNIZER_NAMES:
            raise ConfigurationError(f"unsupported recognizer: {self.recognizer}")
        if self.layout not in LAYOUT_NAMES:
            raise ConfigurationError(f"unsupported layout model: {self.layout}")


@dataclass
class PipelineOptions:
    input_path: Path
    output_path: Path
    work_root: Path = Path("work")
    model_root: Path = Path("models")
    workflow_defaults: Path | None = None
    device: str = "auto"
    batch_size: int = 1
    num_workers: int = 0
    seed: int = 42
    resume: bool = False
    force: bool = False
    fail_fast: bool = False
    keep_intermediate: bool = False
    config: Path | None = None


def enumerate_pipeline_combinations() -> list[PipelineSelection]:
    """Return detector-major Cartesian product in stable CLI order."""
    return [
        PipelineSelection(detector, recognizer, layout)
        for detector in DETECTOR_NAMES
        for recognizer in RECOGNIZER_NAMES
        for layout in LAYOUT_NAMES
        if LAYOUT_ADAPTERS[layout].provides_invoice_labels
    ]


def resolve_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("device must be one of: auto, cpu, cuda")
    if device != "auto":
        return device
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
    except (ImportError, OSError):
        torch_cuda = False
    if torch_cuda:
        return "cuda"
    try:
        import paddle

        paddle_cuda = bool(paddle.is_compiled_with_cuda())
    except (ImportError, OSError):
        paddle_cuda = False
    if paddle_cuda:
        return "cuda"
    return "cpu"


def load_pipeline_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise ConfigurationError(f"pipeline config not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigurationError("pipeline config must be a YAML mapping")
    return loaded


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def load_invoice_schema() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    schema_path = project_root / "configs" / "schema" / "invoice.schema.json"
    if not schema_path.is_file():
        raise ConfigurationError(f"canonical schema not found: {schema_path}")
    loaded = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigurationError("canonical schema must contain a JSON object")
    return loaded


def validate_canonical_payload(payload: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator(load_invoice_schema()).validate(payload)


ITEM_LABEL_TO_COLUMN = {
    "LINE_NUMBER": 0,
    "RAW_DESCRIPTION": 1,
    "MEDICINE_NAME": 2,
    "STRENGTH": 3,
    "MANUFACTURER": 4,
    "COUNTRY_OF_MANUFACTURE": 5,
    "ITEM_BID_PACKAGE": 6,
    "ITEM_CONTRACT_REFERENCE": 7,
    "LOT_NUMBER": 8,
    "EXPIRY_DATE": 9,
    "UNIT": 10,
    "QUANTITY": 11,
    "UNIT_PRICE": 12,
    "LINE_AMOUNT": 13,
    "ITEM_VAT_RATE": 14,
    "VAT_AMOUNT": 15,
}


def entities_to_table_cells(entities: list[LabeledEntity]) -> list[TableCell]:
    """Cluster item entities by vertical center while retaining semantic columns."""
    item_entities = [
        entity
        for entity in entities
        if entity.label.removeprefix("B-").removeprefix("I-") in ITEM_LABEL_TO_COLUMN
    ]
    ordered = sorted(item_entities, key=lambda item: (item.bbox.y_min, item.bbox.x_min))
    rows: list[list[LabeledEntity]] = []
    for entity in ordered:
        center = (entity.bbox.y_min + entity.bbox.y_max) / 2
        height = entity.bbox.y_max - entity.bbox.y_min
        if rows:
            row_center = sum((item.bbox.y_min + item.bbox.y_max) / 2 for item in rows[-1]) / len(
                rows[-1]
            )
        else:
            row_center = float("-inf")
        if rows and abs(center - row_center) <= max(2.0, height * 0.7):
            rows[-1].append(entity)
        else:
            rows.append([entity])
    cells: list[TableCell] = []
    for row_index, row in enumerate(rows):
        for entity in row:
            label = entity.label.removeprefix("B-").removeprefix("I-")
            cells.append(
                TableCell(
                    document_id=entity.document_id,
                    source_path=entity.source_path,
                    page_index=entity.page_index,
                    model_name=entity.model_name,
                    model_revision=entity.model_revision,
                    processing_status=entity.processing_status,
                    table_id=f"{entity.document_id}-p{entity.page_index}-medicine",
                    row_index=row_index,
                    column_index=ITEM_LABEL_TO_COLUMN[label],
                    text=entity.text,
                    bbox=entity.bbox,
                    label=label,
                )
            )
    return cells


class PipelineRunner:
    """Run one concrete A -> B -> C selection."""

    def __init__(
        self,
        selection: PipelineSelection,
        options: PipelineOptions,
        detector: DetectorAdapter | None = None,
        recognizer: RecognizerAdapter | None = None,
        layout: LayoutAdapter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.selection = selection
        self.options = options
        self.device = resolve_device(options.device)
        self.logger = logger or logging.getLogger("invoice_ocr.pipeline")
        self.detector = detector or DETECTORS[selection.detector](options.model_root, self.device)
        self.recognizer = recognizer or RECOGNIZERS[selection.recognizer](
            options.model_root, self.device
        )
        self.layout = layout or LAYOUT_ADAPTERS[selection.layout](options.model_root, self.device)
        self.run_id = options.output_path.name
        self.work_dir = options.work_root / self.run_id

    def _prediction_is_valid(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_canonical_payload(payload)
            return True
        except (OSError, ValueError, jsonschema.ValidationError):
            return False

    def _initialize_artifacts(self) -> None:
        self.options.output_path.mkdir(parents=True, exist_ok=True)
        (self.options.output_path / "predictions").mkdir(parents=True, exist_ok=True)
        (self.options.output_path / "logs").mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.options.force:
            for path in (
                self.work_dir / "pages.jsonl",
                self.work_dir / "detections.jsonl",
                self.work_dir / "recognitions.jsonl",
                self.work_dir / "entities.jsonl",
                self.work_dir / "tables.jsonl",
                self.work_dir / "layout_raw.jsonl",
                self.options.output_path / "errors.jsonl",
            ):
                if path.is_file():
                    path.unlink()

    def run(self) -> RunManifest:
        started = time.perf_counter()
        documents = discover_documents(self.options.input_path)
        self._initialize_artifacts()
        defaults = load_workflow_defaults(self.options.workflow_defaults)
        tolerance = Decimal(str(defaults.get("validation_tolerance", "1")))
        required_fields = [str(value) for value in defaults.get("required_fields", [])]
        manifest = RunManifest(
            run_id=self.run_id,
            command="run",
            detector=ModelIdentity(name=self.selection.detector),
            recognizer=ModelIdentity(name=self.selection.recognizer),
            layout=ModelIdentity(name=self.selection.layout),
            input_path=str(self.options.input_path),
            output_path=str(self.options.output_path),
            device=self.device,
            seed=self.options.seed,
            document_count=len(documents),
            settings={
                "batch_size": self.options.batch_size,
                "num_workers": self.options.num_workers,
                "resume": self.options.resume,
                "keep_intermediate": self.options.keep_intermediate,
                "semantic_invoice_output": self.layout.provides_invoice_labels,
            },
        )
        write_json(self.options.output_path / "manifest.json", manifest.model_dump(mode="python"))
        failed = 0
        completed = 0
        for document in documents:
            prediction_path = (
                self.options.output_path
                / "predictions"
                / prediction_relative_path(document.relative_path)
            )
            if self._prediction_is_valid(prediction_path) and not self.options.force:
                self.logger.info("Keeping valid output for %s", document.relative_path)
                completed += 1
                continue
            if prediction_path.exists() and not self.options.force and not self.options.resume:
                raise OutputExistsError(
                    f"output exists but is not a valid resumable prediction: {prediction_path}; "
                    "use --force to replace it"
                )
            try:
                pages = render_document(document, self.work_dir / "rendered" / document.document_id)
                write_jsonl(self.work_dir / "pages.jsonl", pages, append=True)
                invoices = []
                for page in pages:
                    detections = self.detector.detect(page)
                    write_jsonl(self.work_dir / "detections.jsonl", detections, append=True)
                    recognitions = self.recognizer.recognize(page, detections)
                    write_jsonl(self.work_dir / "recognitions.jsonl", recognitions, append=True)
                    entities, _relations = self.layout.extract(page, recognitions)
                    write_jsonl(self.work_dir / "entities.jsonl", entities, append=True)
                    trace = self.layout.raw_trace()
                    if trace is not None:
                        write_jsonl(self.work_dir / "layout_raw.jsonl", [trace], append=True)
                    if not self.layout.provides_invoice_labels:
                        continue
                    table_cells = entities_to_table_cells(entities)
                    write_jsonl(self.work_dir / "tables.jsonl", table_cells, append=True)
                    invoice = entities_to_invoice(
                        page.page_index + 1, entities, workflow_defaults=defaults
                    )
                    rows = reconstruct_rows(table_cells)
                    invoice.items = [
                        reconstruct_medicine_item(row, line_number=index)
                        for index, row in enumerate(rows, start=1)
                    ]
                    validate_invoice(invoice, tolerance, required_fields)
                    invoices.append(invoice)
                if self.layout.provides_invoice_labels:
                    batch = InvoiceBatch(invoice_count=len(invoices), invoices=invoices)
                    payload = batch.model_dump(mode="python", exclude_none=False)
                    validate_canonical_payload(payload)
                    write_json(prediction_path, payload)
                completed += 1
            except Exception as exc:
                failed += 1
                error = DocumentError(
                    document_id=document.document_id,
                    source_path=document.source_path,
                    stage="pipeline",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    recoverable=not self.options.fail_fast,
                )
                write_jsonl(self.options.output_path / "errors.jsonl", [error], append=True)
                self.logger.error("%s failed: %s", document.relative_path, exc)
                if self.options.fail_fast:
                    raise
        manifest.failed_document_count = failed
        manifest.status = (
            ProcessingStatus.SUCCESS if completed > 0 and failed == 0 else ProcessingStatus.FAILED
        )
        manifest.completed_at = datetime.now(timezone.utc)
        write_json(self.options.output_path / "manifest.json", manifest.model_dump(mode="python"))
        elapsed = time.perf_counter() - started
        metrics = {
            "documents_total": len(documents),
            "documents_completed": completed,
            "failed_document_count": failed,
            "processing_seconds": elapsed,
            "seconds_per_document": elapsed / len(documents),
            "semantic_invoice_output": self.layout.provides_invoice_labels,
        }
        write_json(self.options.output_path / "metrics.json", metrics)
        summary = (
            f"# Run {self.run_id}\n\n"
            f"- Pipeline: `{self.selection.detector} -> {self.selection.recognizer} "
            f"-> {self.selection.layout}`\n"
            f"- Documents: {len(documents)}\n"
            f"- Completed: {completed}\n"
            f"- Failed: {failed}\n"
            f"- Runtime: {elapsed:.3f} seconds\n"
        )
        (self.options.output_path / "summary.md").write_text(summary, encoding="utf-8")
        return manifest

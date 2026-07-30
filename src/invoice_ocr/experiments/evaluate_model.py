"""Locked-split evaluation for one production model adapter."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from invoice_ocr.adapters.detectors import DETECTORS, DetectorAdapter
from invoice_ocr.adapters.layout import LAYOUT_ADAPTERS, LayoutAdapter
from invoice_ocr.adapters.recognizers import RECOGNIZERS, RecognizerAdapter
from invoice_ocr.contracts import (
    BoundingBox,
    DetectionRegion,
    DocumentPage,
    Point,
    ProcessingStatus,
    RecognizedRegion,
)
from invoice_ocr.evaluation.detection import detection_metrics
from invoice_ocr.evaluation.layout import set_metrics
from invoice_ocr.evaluation.recognition import recognition_metrics
from invoice_ocr.exceptions import (
    AnnotationUnavailableError,
    ConfigurationError,
    InvoiceOCRError,
)
from invoice_ocr.experiments.contracts import AggregateMetric
from invoice_ocr.experiments.hashing import (
    canonical_json_hash,
    directory_manifest_hash,
    file_hash_or_missing,
    reproducibility_metadata,
    sha256_file,
)
from invoice_ocr.experiments.runtime import EvaluationTimer
from invoice_ocr.experiments.split import assert_locked_dataset_matches, load_locked_split
from invoice_ocr.io.paths import discover_documents
from invoice_ocr.io.pdf_render import render_document
from invoice_ocr.model_manifest import load_adapter_manifest
from invoice_ocr.pipeline import resolve_device

Stage = Literal["detector", "recognizer", "layout"]
SplitName = Literal["train", "validation", "test"]


@dataclass
class ModelEvaluationRequest:
    stage: Stage
    model: str
    checkpoint_source: str
    data_root: Path
    gt_root: Path
    split_manifest: Path
    output_dir: Path
    checkpoint: Path | None = None
    split: SplitName = "test"
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
    baseline_mode: str | None = None
    finetuned_mode: str | None = None
    validation_tolerance: str = "0.01"
    preprocessing_config: dict[str, Any] | None = None
    decoding_config: dict[str, Any] | None = None
    postprocessing_config: dict[str, Any] | None = None


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


def _bbox(raw: dict[str, Any]) -> BoundingBox:
    bbox = raw.get("bbox")
    if isinstance(bbox, dict):
        return BoundingBox.model_validate(bbox)
    if isinstance(bbox, list) and len(bbox) == 4:
        return BoundingBox(
            x_min=float(bbox[0]),
            y_min=float(bbox[1]),
            x_max=float(bbox[2]),
            y_max=float(bbox[3]),
        )
    polygon = raw.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("annotation region needs bbox or polygon")
    x_values = [float(point["x"] if isinstance(point, dict) else point[0]) for point in polygon]
    y_values = [float(point["y"] if isinstance(point, dict) else point[1]) for point in polygon]
    return BoundingBox(
        x_min=min(x_values),
        y_min=min(y_values),
        x_max=max(x_values),
        y_max=max(y_values),
    )


def _polygon(box: BoundingBox) -> list[Point]:
    return [
        Point(x=box.x_min, y=box.y_min),
        Point(x=box.x_max, y=box.y_min),
        Point(x=box.x_max, y=box.y_max),
        Point(x=box.x_min, y=box.y_max),
    ]


def _annotation_paths(gt_root: Path, stage: Stage, document_ids: set[str]) -> list[Path]:
    directory = {
        "detector": "detection",
        "recognizer": "recognition",
        "layout": "layout",
    }[stage]
    paths = [
        path for path in sorted((gt_root / directory).glob("*.json")) if path.stem in document_ids
    ]
    if paths:
        return paths
    requirement = {
        "detector": "text polygons or bounding boxes in GT/detection/<document_id>.json",
        "recognizer": (
            "source page/crop boxes and exact transcription in GT/recognition/<document_id>.json"
        ),
        "layout": ("OCR tokens, bounding boxes, and entity labels in GT/layout/<document_id>.json"),
    }[stage]
    raise AnnotationUnavailableError(
        f"cannot evaluate {stage} on locked {len(document_ids)}-document split: "
        f"missing {requirement}"
    )


def partition_annotation_paths(
    gt_root: Path,
    stage: Stage,
    train_document_ids: set[str],
    validation_document_ids: set[str],
    test_document_ids: set[str],
) -> tuple[list[Path], list[Path]]:
    """Return train/validation paths while asserting the locked test set is excluded."""
    all_ids = train_document_ids | validation_document_ids | test_document_ids
    if (
        train_document_ids & validation_document_ids
        or train_document_ids & test_document_ids
        or validation_document_ids & test_document_ids
    ):
        raise ValueError("train, validation, and test IDs overlap")
    paths = _annotation_paths(gt_root, stage, all_ids)
    train = [path for path in paths if path.stem in train_document_ids]
    validation = [path for path in paths if path.stem in validation_document_ids]
    if any(path.stem in test_document_ids for path in train + validation):
        raise RuntimeError("locked test annotation leaked into a training dataset")
    return train, validation


def _stage_adapter(request: ModelEvaluationRequest, device: str) -> Any:
    checkpoint = request.checkpoint
    if request.stage == "detector":
        detector_type = DETECTORS.get(request.model)
        if detector_type is None:
            raise ConfigurationError(f"unsupported detector model: {request.model}")
        return detector_type(request.model_root, device, checkpoint)
    if request.stage == "recognizer":
        recognizer_type = RECOGNIZERS.get(request.model)
        if recognizer_type is None:
            raise ConfigurationError(f"unsupported recognizer model: {request.model}")
        return recognizer_type(request.model_root, device, checkpoint)
    layout_type = LAYOUT_ADAPTERS.get(request.model)
    if layout_type is None:
        raise ConfigurationError(f"unsupported layout model: {request.model}")
    return layout_type(request.model_root, device, checkpoint)


def _layout_pretrained_guard(request: ModelEvaluationRequest) -> None:
    if request.stage != "layout" or request.checkpoint_source != "pretrained":
        return
    if request.baseline_mode == "generic_kie_checkpoint":
        manifest = load_adapter_manifest("layout", request.model)
        if manifest.get("expected_task") != "invoice_token_classification":
            raise AnnotationUnavailableError(
                "generic KIE checkpoint is N/A: the official manifest does not declare an "
                "invoice-compatible label space; arbitrary label mapping is forbidden"
            )
        return
    raise AnnotationUnavailableError(
        f"{request.model} base encoder cannot be reported as a pretrained invoice extractor: "
        "a random invoice task head has no pretrained performance. Train a linear_probe head "
        "on train/validation first, then evaluate that checkpoint on the locked test split."
    )


def _checkpoint_identity(request: ModelEvaluationRequest) -> dict[str, Any]:
    manifest = load_adapter_manifest(request.stage, request.model)
    checkpoint_path = request.checkpoint
    digest: str | None = None
    if checkpoint_path is not None:
        if checkpoint_path.is_file():
            digest = sha256_file(checkpoint_path)
        elif checkpoint_path.is_dir():
            digest = directory_manifest_hash(checkpoint_path)
    else:
        local = request.model_root / str(manifest.get("local_path", ""))
        if local.is_file():
            digest = sha256_file(local)
        elif local.is_dir():
            digest = directory_manifest_hash(local)
    return {
        "identifier": manifest.get("checkpoint_identifier"),
        "source_repository": manifest.get("official_repository"),
        "revision": manifest.get("checkpoint_revision") or manifest.get("revision"),
        "source_revision": manifest.get("revision"),
        "path": str(checkpoint_path) if checkpoint_path else None,
        "sha256": digest or manifest.get("sha256"),
        "best_epoch": _best_epoch(checkpoint_path),
    }


def _best_epoch(checkpoint: Path | None) -> int | None:
    if checkpoint is None:
        return None
    metadata = checkpoint / "checkpoint_selection.json"
    if not metadata.is_file():
        metadata = checkpoint.parent / "checkpoint_selection.json"
    if not metadata.is_file():
        return None
    value = _load_object(metadata).get("best_epoch")
    return int(value) if isinstance(value, (int, float)) else None


def _base_manifest(
    request: ModelEvaluationRequest,
    split_hash: str,
    test_ids: list[str],
    device: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    reproducibility = reproducibility_metadata()
    preprocessing = request.preprocessing_config or {
        "renderer": "invoice_ocr.io.pdf_render",
        "orientation": "exif",
        "deskew": "configured_runtime",
    }
    postprocessing = request.postprocessing_config or {"version": "canonical-v1"}
    decoding = request.decoding_config or {"backend_defaults": True}
    schema_path = Path(__file__).resolve().parents[3] / "configs/schema/invoice.schema.json"
    schema_hash = file_hash_or_missing(schema_path)
    defaults_hash = (
        file_hash_or_missing(request.workflow_defaults)
        if request.workflow_defaults is not None
        else canonical_json_hash({"workflow_defaults": None})
    )
    return {
        "experiment_id": request.output_dir.name,
        "run_kind": "finetuned" if request.checkpoint is not None else "pretrained",
        "stage": request.stage,
        "model": request.model,
        "pipeline": {"detector": None, "recognizer": None, "layout": None},
        "baseline_mode": request.baseline_mode,
        "finetuned_mode": request.finetuned_mode,
        "checkpoint_source": request.checkpoint_source,
        "checkpoint": checkpoint,
        "split_manifest_hash": split_hash,
        "test_document_ids": test_ids,
        "schema_version": schema_hash,
        "preprocessing_config": preprocessing,
        "preprocessing_config_hash": canonical_json_hash(preprocessing),
        "decoding_config": decoding,
        "postprocessing_config": postprocessing,
        "postprocessing_config_hash": canonical_json_hash(postprocessing),
        "workflow_defaults_hash": defaults_hash,
        "validation_tolerance": request.validation_tolerance,
        "batch_size": request.batch_size,
        "num_workers": request.num_workers,
        "device": device,
        "hardware_fingerprint": reproducibility["hardware_fingerprint"],
        "metric_code_version": "invoice-ocr-metrics-v2",
        "reproducibility": reproducibility,
        "random_seed": request.seed,
        "command_line": sys.argv,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": None,
        "status": "pending",
        "skip_reason": None,
    }


def _page_annotations(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pages = payload.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            index = int(page.get("page_index", 0))
            regions = page.get("regions", [])
            if isinstance(regions, list):
                result[index].extend(region for region in regions if isinstance(region, dict))
    return result


def _recognition_annotations(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result = _page_annotations(payload)
    regions = payload.get("regions")
    if isinstance(regions, list):
        for region in regions:
            if isinstance(region, dict):
                result[int(region.get("page_index", 0))].append(region)
    return result


def _render_selected_pages(
    documents: dict[str, Any],
    annotations: list[Path],
    work_dir: Path,
    timer: EvaluationTimer,
) -> tuple[dict[tuple[str, int], DocumentPage], int]:
    pages: dict[tuple[str, int], DocumentPage] = {}
    with timer.stage("preprocessing"):
        for annotation in annotations:
            payload = _load_object(annotation)
            document_id = str(payload.get("document_id", annotation.stem))
            document = documents.get(document_id)
            if document is None:
                raise ValueError(
                    f"annotation {annotation} references document_id absent from locked data: "
                    f"{document_id}"
                )
            for page in render_document(document, work_dir / "pages" / document_id):
                pages[(document_id, page.page_index)] = page
    return pages, len(pages)


def _evaluate_detector(
    adapter: DetectorAdapter,
    annotations: list[Path],
    pages: dict[tuple[str, int], DocumentPage],
    timer: EvaluationTimer,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, float]]]:
    total_tp = total_fp = total_fn = 0
    iou_sum = 0.0
    matched_page_count = 0
    evaluated_pages = 0
    records: list[dict[str, Any]] = []
    per_document_rows: dict[str, list[float]] = defaultdict(list)
    for annotation in annotations:
        payload = _load_object(annotation)
        document_id = str(payload.get("document_id", annotation.stem))
        for page_index, expected_regions in _page_annotations(payload).items():
            page = pages[(document_id, page_index)]
            with timer.stage("detection"):
                predicted = adapter.detect(page)
            expected = [_bbox(region) for region in expected_regions]
            metric = detection_metrics([region.bbox for region in predicted], expected)
            total_tp += metric.true_positives
            total_fp += metric.false_positives
            total_fn += metric.false_negatives
            if metric.true_positives:
                iou_sum += metric.mean_iou
                matched_page_count += 1
            evaluated_pages += 1
            per_document_rows[document_id].append(metric.f1)
            records.extend(region.model_dump(mode="json") for region in predicted)
    precision_denominator = total_tp + total_fp
    recall_denominator = total_tp + total_fn
    precision = total_tp / precision_denominator if precision_denominator else 0.0
    recall = total_tp / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_document = {
        document_id: {"detection_f1": sum(values) / len(values)}
        for document_id, values in per_document_rows.items()
    }
    metrics = {
        "detection_precision": _metric(
            precision, total_tp, precision_denominator, evaluated_pages, 0, False
        ),
        "detection_recall": _metric(
            recall, total_tp, recall_denominator, evaluated_pages, 0, False
        ),
        "detection_f1": _metric(f1, None, None, evaluated_pages, 0, False),
        "detection_hmean": _metric(f1, None, None, evaluated_pages, 0, False),
        "detection_mean_iou": _metric(
            iou_sum / matched_page_count if matched_page_count else 0.0,
            iou_sum,
            matched_page_count,
            evaluated_pages,
            0,
            False,
        ),
    }
    return metrics, records, per_document


def _detection_from_gt(page: DocumentPage, region: dict[str, Any], index: int) -> DetectionRegion:
    box = _bbox(region)
    return DetectionRegion(
        document_id=page.document_id,
        source_path=page.source_path,
        page_index=page.page_index,
        model_name="ground-truth-box",
        model_revision=None,
        processing_status=ProcessingStatus.SUCCESS,
        region_id=str(region.get("region_id", f"gt-{page.page_index}-{index}")),
        polygon=_polygon(box),
        bbox=box,
        confidence=1.0,
    )


def _evaluate_recognizer(
    adapter: RecognizerAdapter,
    annotations: list[Path],
    pages: dict[tuple[str, int], DocumentPage],
    timer: EvaluationTimer,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, float]]]:
    all_predicted: list[str] = []
    all_expected: list[str] = []
    records: list[dict[str, Any]] = []
    per_document: dict[str, dict[str, float]] = {}
    numeric_count = numeric_correct = 0
    for annotation in annotations:
        payload = _load_object(annotation)
        document_id = str(payload.get("document_id", annotation.stem))
        document_predicted: list[str] = []
        document_expected: list[str] = []
        for page_index, expected_regions in _recognition_annotations(payload).items():
            page = pages[(document_id, page_index)]
            regions = [
                _detection_from_gt(page, raw, index) for index, raw in enumerate(expected_regions)
            ]
            with timer.stage("recognition"):
                predicted = adapter.recognize(page, regions)
            indexed = {region.region_id: region for region in predicted}
            for index, raw in enumerate(expected_regions):
                region_id = str(raw.get("region_id", regions[index].region_id))
                prediction = indexed.get(region_id)
                text = prediction.text if prediction is not None else ""
                expected = str(raw["text"])
                document_predicted.append(text)
                document_expected.append(expected)
            records.extend(region.model_dump(mode="json") for region in predicted)
        metric = recognition_metrics(document_predicted, document_expected)
        per_document[document_id] = {"recognition_cer": metric.cer}
        all_predicted.extend(document_predicted)
        all_expected.extend(document_expected)
    aggregate = recognition_metrics(all_predicted, all_expected)
    char_denominator = sum(len(value) for value in all_expected)
    word_denominator = sum(len(value.split()) for value in all_expected)
    exact_correct = sum(
        predicted_text == expected_text
        for predicted_text, expected_text in zip(all_predicted, all_expected, strict=True)
    )
    from invoice_ocr.evaluation.recognition import NUMERIC_PATTERN

    for predicted_text, expected_text in zip(all_predicted, all_expected, strict=True):
        if NUMERIC_PATTERN.fullmatch(expected_text.strip()):
            numeric_count += 1
            numeric_correct += predicted_text == expected_text
    metrics = {
        "recognition_cer": _metric(
            aggregate.cer,
            aggregate.cer * char_denominator,
            char_denominator,
            len(all_expected),
            0,
            True,
        ),
        "recognition_wer": _metric(
            aggregate.wer,
            aggregate.wer * word_denominator,
            word_denominator,
            len(all_expected),
            0,
            True,
        ),
        "recognition_exact_match": _metric(
            aggregate.exact_match,
            exact_correct,
            len(all_expected),
            len(all_expected),
            0,
            False,
        ),
        "recognition_numeric_accuracy": _metric(
            numeric_correct / numeric_count if numeric_count else None,
            numeric_correct if numeric_count else None,
            numeric_count if numeric_count else None,
            numeric_count,
            len(all_expected) - numeric_count,
            False,
            None if numeric_count else "locked split has no numeric transcription samples",
        ),
    }
    return metrics, records, per_document


def _layout_regions(
    page: DocumentPage, tokens: list[Any], boxes: list[Any]
) -> list[RecognizedRegion]:
    regions: list[RecognizedRegion] = []
    for index, (token, raw_box) in enumerate(zip(tokens, boxes, strict=True)):
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError("layout annotation boxes must contain four coordinates")
        normalized = all(isinstance(value, int) and 0 <= value <= 1000 for value in raw_box)
        values = [float(value) for value in raw_box]
        if normalized:
            values = [
                values[0] / 1000 * page.width,
                values[1] / 1000 * page.height,
                values[2] / 1000 * page.width,
                values[3] / 1000 * page.height,
            ]
        box = BoundingBox(x_min=values[0], y_min=values[1], x_max=values[2], y_max=values[3])
        regions.append(
            RecognizedRegion(
                document_id=page.document_id,
                source_path=page.source_path,
                page_index=page.page_index,
                model_name="ground-truth-ocr",
                model_revision=None,
                processing_status=ProcessingStatus.SUCCESS,
                region_id=f"gt-{page.page_index}-{index}",
                polygon=_polygon(box),
                bbox=box,
                text=str(token),
                confidence=1.0,
            )
        )
    return regions


def _evaluate_layout(
    adapter: LayoutAdapter,
    annotations: list[Path],
    pages: dict[tuple[str, int], DocumentPage],
    timer: EvaluationTimer,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, float]]]:
    predicted_set: set[tuple[object, ...]] = set()
    expected_set: set[tuple[object, ...]] = set()
    records: list[dict[str, Any]] = []
    per_document_sets: dict[str, tuple[set[tuple[object, ...]], set[tuple[object, ...]]]] = {}
    evaluated_pages = 0
    has_relations = False
    for annotation in annotations:
        payload = _load_object(annotation)
        document_id = str(payload.get("document_id", annotation.stem))
        local_predicted: set[tuple[object, ...]] = set()
        local_expected: set[tuple[object, ...]] = set()
        has_relations = has_relations or bool(payload.get("relations"))
        for page_payload in payload.get("pages", []):
            page_index = int(page_payload.get("page_index", 0))
            page = pages[(document_id, page_index)]
            tokens = list(page_payload["tokens"])
            labels = list(page_payload["labels"])
            regions = _layout_regions(page, tokens, list(page_payload["boxes"]))
            with timer.stage("layout_inference"):
                entities, relations = adapter.extract(page, regions)
            for entity in entities:
                entry = (
                    document_id,
                    page_index,
                    entity.label,
                    " ".join(entity.text.split()),
                )
                predicted_set.add(entry)
                local_predicted.add(entry)
                records.append(entity.model_dump(mode="json"))
            records.extend(relation.model_dump(mode="json") for relation in relations)
            for token, label in zip(tokens, labels, strict=True):
                if str(label) != "O":
                    entry = (document_id, page_index, str(label), str(token).strip())
                    expected_set.add(entry)
                    local_expected.add(entry)
            evaluated_pages += 1
        per_document_sets[document_id] = (local_predicted, local_expected)
    aggregate = set_metrics(predicted_set, expected_set)
    per_document = {
        document_id: {"entity_f1": set_metrics(predicted, expected).f1}
        for document_id, (predicted, expected) in per_document_sets.items()
    }
    metrics = {
        "entity_precision": _metric(
            aggregate.precision,
            aggregate.true_positives,
            aggregate.true_positives + aggregate.false_positives,
            evaluated_pages,
            0,
            False,
        ),
        "entity_recall": _metric(
            aggregate.recall,
            aggregate.true_positives,
            aggregate.true_positives + aggregate.false_negatives,
            evaluated_pages,
            0,
            False,
        ),
        "entity_f1": _metric(aggregate.f1, None, None, evaluated_pages, 0, False),
        "field_exact_match": _metric(aggregate.f1, None, None, evaluated_pages, 0, False),
        "relation_f1": _metric(
            None,
            None,
            None,
            0,
            evaluated_pages,
            False,
            (
                "predicted relation persistence is not available for this adapter"
                if has_relations
                else "GT/layout relation annotations are missing"
            ),
        ),
    }
    return metrics, records, per_document


def _na_metrics(stage: Stage, reason: str, skipped_count: int) -> dict[str, Any]:
    names = {
        "detector": (
            "detection_precision",
            "detection_recall",
            "detection_f1",
            "detection_hmean",
            "detection_mean_iou",
        ),
        "recognizer": (
            "recognition_cer",
            "recognition_wer",
            "recognition_exact_match",
            "recognition_numeric_accuracy",
        ),
        "layout": (
            "entity_precision",
            "entity_recall",
            "entity_f1",
            "relation_f1",
            "field_exact_match",
        ),
    }[stage]
    return {
        name: _metric(
            None,
            None,
            None,
            0,
            skipped_count,
            any(token in name for token in ("cer", "wer")),
            reason,
        )
        for name in names
    }


def _write_config(request: ModelEvaluationRequest, selected_ids: list[str]) -> None:
    payload = {
        "stage": request.stage,
        "model": request.model,
        "checkpoint_source": request.checkpoint_source,
        "checkpoint": str(request.checkpoint) if request.checkpoint else None,
        "split": request.split,
        "selected_document_ids": selected_ids,
        "device": request.device,
        "batch_size": request.batch_size,
        "num_workers": request.num_workers,
        "warmup_iterations": request.warmup_iterations,
        "seed": request.seed,
    }
    (request.output_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def _resume_complete(output_dir: Path) -> bool:
    required = ("manifest.json", "metrics.json", "timing.json")
    if not all((output_dir / name).is_file() for name in required):
        return False
    manifest = _load_object(output_dir / "manifest.json")
    return manifest.get("status") == "success"


def _warm_up(
    adapter: DetectorAdapter | RecognizerAdapter | LayoutAdapter,
    annotations: list[Path],
    pages: dict[tuple[str, int], DocumentPage],
    iterations: int,
) -> None:
    if iterations == 0:
        return
    payload = _load_object(annotations[0])
    document_id = str(payload.get("document_id", annotations[0].stem))
    if isinstance(adapter, DetectorAdapter):
        first_page = next(page for key, page in pages.items() if key[0] == document_id)
        for _ in range(iterations):
            adapter.detect(first_page)
        return
    if isinstance(adapter, RecognizerAdapter):
        by_page = _recognition_annotations(payload)
        page_index, raw_regions = next(iter(by_page.items()))
        page = pages[(document_id, page_index)]
        detection_regions = [
            _detection_from_gt(page, raw, index) for index, raw in enumerate(raw_regions)
        ]
        for _ in range(iterations):
            adapter.recognize(page, detection_regions)
        return
    page_payload = next(page for page in payload["pages"] if isinstance(page, dict))
    page_index = int(page_payload.get("page_index", 0))
    page = pages[(document_id, page_index)]
    layout_regions = _layout_regions(
        page, list(page_payload["tokens"]), list(page_payload["boxes"])
    )
    for _ in range(iterations):
        adapter.extract(page, layout_regions)


def evaluate_model(request: ModelEvaluationRequest) -> Path:
    """Evaluate one real adapter without allowing test-set protocol drift."""
    if request.batch_size <= 0 or request.num_workers < 0:
        raise ConfigurationError("batch size must be positive and num workers non-negative")
    if request.output_dir.exists() and request.resume and _resume_complete(request.output_dir):
        return request.output_dir
    if (
        request.output_dir.exists()
        and not request.force
        and not request.resume
        and any(request.output_dir.iterdir())
    ):
        raise ConfigurationError(
            f"evaluation output already exists: {request.output_dir}; use --resume or --force"
        )
    manifest_split = load_locked_split(request.split_manifest)
    assert_locked_dataset_matches(manifest_split, request.data_root, request.gt_root)
    selected_ids = manifest_split.ids_for(request.split)
    if not selected_ids:
        raise AnnotationUnavailableError(f"locked split '{request.split}' contains no document IDs")
    documents = {
        document.document_id: document for document in discover_documents(request.data_root)
    }
    selected_set = set(selected_ids)
    selected_documents = {key: value for key, value in documents.items() if key in selected_set}
    if set(selected_documents) != selected_set:
        missing = sorted(selected_set - set(selected_documents))
        raise ValueError(f"locked split documents are absent from data: {missing}")
    device = resolve_device(request.device)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    (request.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    _write_config(request, selected_ids)
    checkpoint = _checkpoint_identity(request)
    manifest = _base_manifest(
        request,
        manifest_split.split_manifest_hash,
        selected_ids if request.split == "test" else [],
        device,
        checkpoint,
    )
    timer = EvaluationTimer(request.warmup_iterations)
    timer.start()
    try:
        _layout_pretrained_guard(request)
        annotations = _annotation_paths(request.gt_root, request.stage, selected_set)
        with timer.stage("model_load"):
            adapter = _stage_adapter(request, device)
            adapter.prepare()
        pages, page_count = _render_selected_pages(
            selected_documents,
            annotations,
            request.work_root / "experiments" / request.output_dir.name,
            timer,
        )
        _warm_up(adapter, annotations, pages, request.warmup_iterations)
        with timer.stage("evaluation"):
            if isinstance(adapter, DetectorAdapter):
                metrics, records, per_document = _evaluate_detector(
                    adapter, annotations, pages, timer
                )
                primary_metric = "detection_f1"
            elif isinstance(adapter, RecognizerAdapter):
                metrics, records, per_document = _evaluate_recognizer(
                    adapter, annotations, pages, timer
                )
                primary_metric = "recognition_cer"
            else:
                metrics, records, per_document = _evaluate_layout(
                    adapter, annotations, pages, timer
                )
                primary_metric = "entity_f1"
        timing = timer.finish(len(selected_documents), page_count, 0, 0)
        _write_jsonl(request.output_dir / "predictions" / "stage_predictions.jsonl", records)
        _write_object(
            request.output_dir / "metrics.json",
            {
                "metrics": metrics,
                "primary_metric": primary_metric,
                "per_document": per_document,
                "per_field": {},
            },
        )
        _write_object(request.output_dir / "timing.json", timing.model_dump(mode="json"))
        (request.output_dir / "errors.jsonl").write_text("", encoding="utf-8")
        manifest["status"] = "success"
    except InvoiceOCRError as exc:
        timing = timer.finish(0, 0, 0, len(selected_ids))
        reason = str(exc)
        _write_object(
            request.output_dir / "metrics.json",
            {
                "metrics": _na_metrics(request.stage, reason, len(selected_ids)),
                "primary_metric": "",
                "per_document": {},
                "per_field": {},
            },
        )
        _write_object(request.output_dir / "timing.json", timing.model_dump(mode="json"))
        _write_jsonl(
            request.output_dir / "errors.jsonl",
            [{"status": "SKIPPED", "stage": request.stage, "reason": reason}],
        )
        manifest["status"] = "skipped"
        manifest["skip_reason"] = reason
    manifest["end_time"] = datetime.now(timezone.utc).isoformat()
    _write_object(request.output_dir / "manifest.json", manifest)
    return request.output_dir

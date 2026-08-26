"""Run pretrained OCR and serialize pseudo LayoutLM annotations and audit reports."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invoice_ocr.adapters.detectors import DETECTORS, DetectorAdapter
from invoice_ocr.adapters.recognizers import RECOGNIZERS, RecognizerAdapter
from invoice_ocr.contracts import DocumentPage, SourceDocument
from invoice_ocr.exceptions import ConfigurationError, OutputExistsError
from invoice_ocr.io.paths import discover_documents
from invoice_ocr.io.pdf_render import render_document
from invoice_ocr.layout_gt.alignment import (
    PSEUDO_ANNOTATION_SOURCE,
    AlignmentResult,
    OCRRegion,
    align_ground_truth_fields,
    flatten_canonical_ground_truth,
    normalize_layout_bbox,
)
from invoice_ocr.layout_gt.index import (
    FinalGroundTruthIndex,
    build_final_ground_truth_index,
    final_gt_path,
    load_final_ground_truth_index,
)
from invoice_ocr.pipeline import resolve_device
from invoice_ocr.training.datasets import validate_ground_truth

Renderer = Callable[[SourceDocument, Path], list[DocumentPage]]
LOGGER = logging.getLogger("invoice_ocr")


@dataclass(frozen=True)
class LayoutGTBuildRequest:
    input_root: Path
    gt_root: Path
    output_dir: Path
    detector_name: str = "paddleocr"
    recognizer_name: str = "vietocr"
    model_root: Path = Path("models")
    device: str = "auto"
    gt_prefix: str | None = None
    target_manifest: Path | None = None
    force: bool = False
    max_alignment_boxes: int = 12


@dataclass(frozen=True)
class DocumentAlignment:
    document_id: str
    source_relative_path: str
    gt_relative_path: str
    result: AlignmentResult


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


def _prepare_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise OutputExistsError(
                f"pseudo-layout output already exists: {output_dir}; use --force to rebuild"
            )
        for directory in ("images", "layout", "ocr"):
            target = output_dir / directory
            if target.is_dir():
                shutil.rmtree(target)
        for name in (
            "manifest.json",
            "document_index.json",
            "alignment_report.json",
            "alignment_report.csv",
        ):
            target = output_dir / name
            if target.is_file():
                target.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _resolve_index(request: LayoutGTBuildRequest) -> FinalGroundTruthIndex:
    if request.target_manifest is not None:
        index = load_final_ground_truth_index(request.target_manifest)
        if Path(index.data_root).resolve() != request.input_root.expanduser().resolve():
            raise ConfigurationError(
                "target manifest data_root differs from --input; rebuild the target manifest "
                "for this dataset"
            )
        return index
    return build_final_ground_truth_index(
        request.input_root,
        request.gt_root,
        request.output_dir / "document_index.json",
        request.gt_prefix,
    )


def _ocr_regions(
    page: DocumentPage,
    detector: DetectorAdapter,
    recognizer: RecognizerAdapter,
) -> list[OCRRegion]:
    detections = detector.detect(page)
    detection_by_id = {region.region_id: region for region in detections}
    recognized = recognizer.recognize(page, detections)
    result: list[OCRRegion] = []
    for region in recognized:
        if not region.text.strip():
            continue
        detection = detection_by_id.get(region.region_id)
        if detection is None:
            raise ValueError(
                f"recognizer returned unknown region_id {region.region_id} for {page.document_id}"
            )
        result.append(
            OCRRegion(
                region_id=region.region_id,
                page_index=page.page_index,
                text=region.text,
                bbox=region.bbox,
                polygon=[[point.x, point.y] for point in region.polygon],
                detection_confidence=detection.confidence,
                recognition_confidence=region.confidence,
            )
        )
    return sorted(
        result,
        key=lambda region: (region.page_index, region.bbox.y_min, region.bbox.x_min),
    )


def _relative_image_path(page: DocumentPage, output_dir: Path) -> str:
    try:
        return Path(page.image_path).resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"rendered page must be stored below pseudo-layout output: {page.image_path}"
        ) from exc


def _serialize_document(
    output_dir: Path,
    document: SourceDocument,
    gt_relative_path: str,
    pages: list[DocumentPage],
    regions: list[OCRRegion],
    result: AlignmentResult,
) -> None:
    regions_by_page: dict[int, list[OCRRegion]] = defaultdict(list)
    for region in regions:
        regions_by_page[region.page_index].append(region)
    layout_pages: list[dict[str, Any]] = []
    ocr_pages: list[dict[str, Any]] = []
    for page in pages:
        page_regions = sorted(
            regions_by_page[page.page_index],
            key=lambda region: (region.bbox.y_min, region.bbox.x_min, region.region_id),
        )
        serialized_regions = [
            {
                "region_id": region.region_id,
                "text": region.text,
                "bbox": [
                    region.bbox.x_min,
                    region.bbox.y_min,
                    region.bbox.x_max,
                    region.bbox.y_max,
                ],
                "normalized_bbox": normalize_layout_bbox(region.bbox, page.width, page.height),
                "polygon": region.polygon,
                "detection_confidence": region.detection_confidence,
                "recognition_confidence": region.recognition_confidence,
                "label": result.region_labels.get(region.region_id, "O"),
                "source": PSEUDO_ANNOTATION_SOURCE,
            }
            for region in page_regions
        ]
        ocr_pages.append(
            {
                "page_index": page.page_index,
                "page": page.page_index + 1,
                "image_path": _relative_image_path(page, output_dir),
                "width": page.width,
                "height": page.height,
                "regions": serialized_regions,
            }
        )
        if page_regions:
            layout_pages.append(
                {
                    "page_index": page.page_index,
                    "page": page.page_index + 1,
                    "image_path": _relative_image_path(page, output_dir),
                    "width": page.width,
                    "height": page.height,
                    "tokens": [region.text for region in page_regions],
                    "boxes": [
                        normalize_layout_bbox(region.bbox, page.width, page.height)
                        for region in page_regions
                    ],
                    "labels": [
                        result.region_labels.get(region.region_id, "O") for region in page_regions
                    ],
                    "region_ids": [region.region_id for region in page_regions],
                    "regions": serialized_regions,
                }
            )
    _write_object(
        output_dir / "ocr" / f"{document.document_id}.json",
        {
            "document_id": document.document_id,
            "source_relative_path": document.relative_path,
            "source": PSEUDO_ANNOTATION_SOURCE,
            "pages": ocr_pages,
        },
    )
    if not layout_pages:
        return
    _write_object(
        output_dir / "layout" / f"{document.document_id}.json",
        {
            "document_id": document.document_id,
            "source_relative_path": document.relative_path,
            "gt_relative_path": gt_relative_path,
            "annotation_kind": "pseudo_layout_gt",
            "source": PSEUDO_ANNOTATION_SOURCE,
            "pages": layout_pages,
            "alignments": [match.as_dict() for match in result.matches],
            "unmatched_fields": [field.as_dict() for field in result.unmatched],
            "relations": [],
        },
    )


def _method_counts(matches: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "exact_matches": sum(
            str(match["match_method"]) in {"exact", "multi_box_exact"} for match in matches
        ),
        "normalized_matches": sum("normalized" in str(match["match_method"]) for match in matches),
        "multi_box_matches": sum(len(match.get("region_ids", [])) > 1 for match in matches),
        "ambiguous_matches": sum(bool(match["ambiguous"]) for match in matches),
        "duplicate_candidate_matches": sum(
            bool(match["duplicate_candidates"]) for match in matches
        ),
    }


def _group_coverage(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    totals: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    eligible: Counter[str] = Counter()
    ambiguous: Counter[str] = Counter()
    for row in rows:
        name = str(row[key])
        totals[name] += 1
        if row["status"] == "matched":
            matched[name] += 1
        if row.get("training_eligible") is True:
            eligible[name] += 1
        if row.get("ambiguous") is True:
            ambiguous[name] += 1
    return {
        name: {
            "total_gt_fields": total,
            "matched_fields": matched[name],
            "unmatched_fields": total - matched[name],
            "ambiguous_matches": ambiguous[name],
            "training_eligible_fields": eligible[name],
            "coverage_percent": matched[name] / total * 100 if total else 0.0,
            "training_coverage_percent": eligible[name] / total * 100 if total else 0.0,
        }
        for name, total in sorted(totals.items())
    }


def write_alignment_report(
    output_dir: Path,
    documents: list[DocumentAlignment],
    index: FinalGroundTruthIndex,
    document_errors: list[dict[str, str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    match_payloads: list[dict[str, Any]] = []
    for document in documents:
        match_by_path = {
            match.field.field_path: match.as_dict() for match in document.result.matches
        }
        for target in document.result.fields:
            match = match_by_path.get(target.field_path)
            if match is None:
                rows.append(
                    {
                        "document_id": document.document_id,
                        "source_relative_path": document.source_relative_path,
                        "gt_relative_path": document.gt_relative_path,
                        "field_path": target.field_path,
                        "label": target.label,
                        "gt_value": target.gt_value,
                        "status": "unmatched",
                        "ocr_text": "",
                        "page": target.page_index + 1 if target.page_index is not None else "",
                        "match_method": "unmatched",
                        "match_confidence": 0.0,
                        "ambiguous": False,
                        "duplicate_candidates": False,
                        "candidate_count": 0,
                        "training_eligible": False,
                    }
                )
                continue
            match_payloads.append(match)
            rows.append(
                {
                    "document_id": document.document_id,
                    "source_relative_path": document.source_relative_path,
                    "gt_relative_path": document.gt_relative_path,
                    "field_path": target.field_path,
                    "label": target.label,
                    "gt_value": target.gt_value,
                    "status": "matched",
                    "ocr_text": match["ocr_text"],
                    "page": match["page"],
                    "match_method": match["match_method"],
                    "match_confidence": match["match_confidence"],
                    "ambiguous": match["ambiguous"],
                    "duplicate_candidates": match["duplicate_candidates"],
                    "candidate_count": match["candidate_count"],
                    "training_eligible": match["training_eligible"],
                }
            )
    total = len(rows)
    matched = sum(row["status"] == "matched" for row in rows)
    eligible = sum(row["training_eligible"] is True for row in rows)
    summary: dict[str, Any] = {
        "total_gt_fields": total,
        "matched_fields": matched,
        "unmatched_fields": total - matched,
        **_method_counts(match_payloads),
        "training_eligible_fields": eligible,
        "coverage_percent": matched / total * 100 if total else 0.0,
        "training_coverage_percent": eligible / total * 100 if total else 0.0,
        "target_documents": index.target_count,
        "processed_documents": len(documents),
        "failed_documents": len(document_errors),
        "excluded_original_or_non_target_json": index.excluded_gt_count,
    }
    report = {
        "schema_version": "layout-alignment-report-v1",
        "source": PSEUDO_ANNOTATION_SOURCE,
        "detector_iou": {
            "status": "N/A",
            "value": None,
            "reason": (
                "human detector GT is not available; pretrained detector boxes are pseudo-layout "
                "annotations and are never used to benchmark the detector"
            ),
        },
        "summary": summary,
        "coverage_by_field": _group_coverage(rows, "label"),
        "coverage_by_document": _group_coverage(rows, "document_id"),
        "fields": rows,
        "document_errors": document_errors,
    }
    _write_object(output_dir / "alignment_report.json", report)
    columns = [
        "document_id",
        "source_relative_path",
        "gt_relative_path",
        "field_path",
        "label",
        "gt_value",
        "status",
        "ocr_text",
        "page",
        "match_method",
        "match_confidence",
        "ambiguous",
        "duplicate_candidates",
        "candidate_count",
        "training_eligible",
    ]
    with (output_dir / "alignment_report.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return report


def build_layout_ground_truth(
    request: LayoutGTBuildRequest,
    *,
    detector: DetectorAdapter | None = None,
    recognizer: RecognizerAdapter | None = None,
    renderer: Renderer = render_document,
) -> Path:
    """Build runtime-only LayoutLM pseudo-GT from pretrained detector/recognizer output."""
    if request.max_alignment_boxes <= 0:
        raise ConfigurationError("max alignment boxes must be positive")
    _prepare_output(request.output_dir, request.force)
    index = _resolve_index(request)
    total_documents = index.target_count
    LOGGER.info(
        "Starting pseudo-layout GT build: targets=%d detector=%s recognizer=%s",
        total_documents,
        request.detector_name,
        request.recognizer_name,
    )
    # Persist a local copy even when an external target manifest was supplied.
    _write_object(request.output_dir / "document_index.json", index.model_dump(mode="json"))
    documents = {
        document.document_id: document for document in discover_documents(request.input_root)
    }
    missing_documents = sorted(set(index.by_document_id()) - set(documents))
    if missing_documents:
        raise ConfigurationError(
            f"target manifest references source documents absent from --input: {missing_documents}"
        )
    device = resolve_device(request.device)
    detector_instance = detector
    if detector_instance is None:
        detector_type = DETECTORS.get(request.detector_name)
        if detector_type is None:
            raise ConfigurationError(f"unsupported detector: {request.detector_name}")
        detector_instance = detector_type(request.model_root, device)
    recognizer_instance = recognizer
    if recognizer_instance is None:
        recognizer_type = RECOGNIZERS.get(request.recognizer_name)
        if recognizer_type is None:
            raise ConfigurationError(f"unsupported recognizer: {request.recognizer_name}")
        recognizer_instance = recognizer_type(request.model_root, device)
    LOGGER.info("Preparing detector on %s: %s", device, request.detector_name)
    detector_instance.prepare()
    LOGGER.info("Detector ready; preparing recognizer: %s", request.recognizer_name)
    recognizer_instance.prepare()
    LOGGER.info("Recognizer ready; starting OCR and GT alignment")
    alignments: list[DocumentAlignment] = []
    errors: list[dict[str, str]] = []
    for position, target in enumerate(index.documents, start=1):
        document = documents[target.document_id]
        started_at = time.perf_counter()
        LOGGER.info(
            "[%d/%d] Processing %s",
            position,
            total_documents,
            document.relative_path,
        )
        try:
            pages = renderer(document, request.output_dir / "images" / document.document_id)
            regions = [
                region
                for page in pages
                for region in _ocr_regions(page, detector_instance, recognizer_instance)
            ]
            gt_payload = _load_object(final_gt_path(index, target))
            fields = flatten_canonical_ground_truth(gt_payload)
            result = align_ground_truth_fields(
                fields,
                regions,
                max_boxes=request.max_alignment_boxes,
            )
            _serialize_document(
                request.output_dir,
                document,
                target.gt_relative_path,
                pages,
                regions,
                result,
            )
            if not regions:
                errors.append(
                    {
                        "document_id": document.document_id,
                        "source_relative_path": document.relative_path,
                        "reason": "pretrained OCR returned no non-empty text regions",
                    }
                )
            alignments.append(
                DocumentAlignment(
                    document_id=document.document_id,
                    source_relative_path=document.relative_path,
                    gt_relative_path=target.gt_relative_path,
                    result=result,
                )
            )
            LOGGER.info(
                "[%d/%d] Completed %s: pages=%d regions=%d matched=%d "
                "unmatched=%d ambiguous=%d elapsed=%.1fs",
                position,
                total_documents,
                document.relative_path,
                len(pages),
                len(regions),
                len(result.matches),
                len(result.unmatched),
                sum(match.ambiguous for match in result.matches),
                time.perf_counter() - started_at,
            )
        except Exception as exc:
            errors.append(
                {
                    "document_id": document.document_id,
                    "source_relative_path": document.relative_path,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            LOGGER.error(
                "[%d/%d] Failed %s after %.1fs: %s: %s",
                position,
                total_documents,
                document.relative_path,
                time.perf_counter() - started_at,
                type(exc).__name__,
                exc,
            )
    report = write_alignment_report(request.output_dir, alignments, index, errors)
    _write_object(
        request.output_dir / "manifest.json",
        {
            "schema_version": "pseudo-layout-gt-v1",
            "annotation_kind": "pseudo_layout_gt",
            "source": PSEUDO_ANNOTATION_SOURCE,
            "input_root": str(request.input_root.expanduser().resolve()),
            "gt_root": str(request.gt_root.expanduser().resolve()),
            "gt_prefix": index.gt_prefix,
            "detector": request.detector_name,
            "recognizer": request.recognizer_name,
            "target_count": index.target_count,
            "layout_annotation_count": len(list((request.output_dir / "layout").glob("*.json"))),
            "failed_document_count": len(errors),
            "alignment_summary": report["summary"],
            "detector_iou": report["detector_iou"],
        },
    )
    summary = report["summary"]
    LOGGER.info(
        "Pseudo-layout GT build completed: processed=%d failed=%d matched=%d/%d "
        "coverage=%.2f%% output=%s",
        summary["processed_documents"],
        summary["failed_documents"],
        summary["matched_fields"],
        summary["total_gt_fields"],
        summary["coverage_percent"],
        request.output_dir,
    )
    return request.output_dir


def inspect_layout_ground_truth(layout_gt_root: Path) -> dict[str, Any]:
    report_path = layout_gt_root / "alignment_report.json"
    manifest_path = layout_gt_root / "manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"pseudo-layout manifest/report not found under {layout_gt_root}; run build-layout-gt"
        )
    report = _load_object(report_path)
    validation = validate_ground_truth(layout_gt_root)
    return {
        "root": str(layout_gt_root.expanduser().resolve()),
        "valid_layout_dataset": validation.is_valid,
        "layout_annotation_count": validation.layout_count,
        "validation_errors": validation.errors,
        "summary": report.get("summary", {}),
        "detector_iou": report.get("detector_iou"),
    }

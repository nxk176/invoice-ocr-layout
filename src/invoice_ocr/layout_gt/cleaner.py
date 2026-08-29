"""Build a conservative, auditable training view of pseudo-layout annotations."""

from __future__ import annotations

import copy
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from invoice_ocr.exceptions import ConfigurationError, OutputExistsError
from invoice_ocr.training.datasets import validate_ground_truth

LOGGER = logging.getLogger("invoice_ocr")

CleaningDecision = Literal["KEEP", "REVIEW", "IGNORE"]

CONSERVATIVE_SAFE_METHODS = frozenset(
    {
        "exact",
        "whitespace_normalized",
        "exact_normalized",
        "anchored_exact_normalized",
        "identifier_normalized",
        "money_normalized",
        "number_normalized",
        "date_normalized",
    }
)


@dataclass(frozen=True)
class LayoutGTCleanRequest:
    """Parameters for deriving a clean dataset without changing the OCR cache."""

    layout_gt_root: Path
    output_dir: Path
    profile: str = "conservative"


@dataclass
class _AlignmentDecision:
    alignment: dict[str, Any]
    decision: CleaningDecision
    reason: str


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _prepare_paths(request: LayoutGTCleanRequest) -> tuple[Path, Path]:
    source = request.layout_gt_root.expanduser().resolve()
    output = request.output_dir.expanduser().resolve()
    if request.profile != "conservative":
        raise ConfigurationError(f"unsupported layout cleaning profile: {request.profile}")
    if not (source / "manifest.json").is_file() or not (source / "layout").is_dir():
        raise FileNotFoundError(
            f"pseudo-layout manifest/layout not found under {source}; run build-layout-gt first"
        )
    if output == source or source in output.parents:
        raise ConfigurationError("clean output must be outside the source pseudo-layout directory")
    if output.exists() and any(output.iterdir()):
        raise OutputExistsError(
            f"clean layout output already exists: {output}; choose a new --output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    return source, output


def _base_method(method: str) -> str:
    return method.removeprefix("multi_box_")


def _float_value(alignment: dict[str, Any], key: str) -> float:
    value = alignment.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _classify_alignment(alignment: dict[str, Any]) -> _AlignmentDecision:
    raw_region_ids = alignment.get("region_ids")
    if not isinstance(raw_region_ids, list) or not raw_region_ids:
        return _AlignmentDecision(alignment, "IGNORE", "alignment_has_no_regions")
    if len(raw_region_ids) != len(set(str(value) for value in raw_region_ids)):
        return _AlignmentDecision(alignment, "IGNORE", "alignment_repeats_region")
    if alignment.get("ambiguous") is True:
        return _AlignmentDecision(alignment, "REVIEW", "ambiguous_match")
    if alignment.get("training_eligible") is not True:
        return _AlignmentDecision(alignment, "REVIEW", "source_not_training_eligible")
    if (
        alignment.get("duplicate_candidates") is True
        or int(alignment.get("candidate_count", 0)) > 1
    ):
        return _AlignmentDecision(alignment, "REVIEW", "duplicate_candidates")
    if len(raw_region_ids) > 1:
        return _AlignmentDecision(alignment, "REVIEW", "multi_box_match")
    method = str(alignment.get("match_method", ""))
    if _base_method(method) not in CONSERVATIVE_SAFE_METHODS:
        return _AlignmentDecision(alignment, "REVIEW", f"method_requires_review:{method}")
    if _float_value(alignment, "match_confidence") < 0.98:
        return _AlignmentDecision(alignment, "REVIEW", "low_match_confidence")
    if _float_value(alignment, "recognition_confidence") < 0.80:
        return _AlignmentDecision(alignment, "REVIEW", "low_recognition_confidence")
    if _float_value(alignment, "detection_confidence") < 0.80:
        return _AlignmentDecision(alignment, "REVIEW", "low_detection_confidence")
    return _AlignmentDecision(alignment, "KEEP", "safe_exact_or_typed_normalized_match")


def _relative_source_image(source: Path, output: Path, image_value: Any) -> str:
    if not isinstance(image_value, str) or not image_value:
        raise ValueError("cleaned layout page requires a non-empty image_path")
    image = (source / image_value).resolve()
    try:
        image.relative_to(source)
    except ValueError as exc:
        raise ValueError(f"source layout image escapes dataset root: {image_value}") from exc
    if not image.is_file():
        raise FileNotFoundError(f"source layout image not found: {image}")
    return Path(os.path.relpath(image, output)).as_posix()


def _page_region_map(pages: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for page_index, page in enumerate(pages):
        tokens = page.get("tokens")
        region_ids = page.get("region_ids")
        if not isinstance(tokens, list) or not isinstance(region_ids, list):
            raise ValueError("cleaning requires token and region_ids arrays on every layout page")
        if len(tokens) != len(region_ids):
            raise ValueError("layout tokens and region_ids must have equal lengths")
        for token_index, raw_region_id in enumerate(region_ids):
            region_id = str(raw_region_id)
            if region_id in result:
                raise ValueError(f"layout document repeats region_id: {region_id}")
            result[region_id] = (page_index, token_index)
    return result


def _resolve_conflicts(
    decisions: list[_AlignmentDecision],
    locations: dict[str, tuple[int, int]],
) -> None:
    by_region: dict[str, list[_AlignmentDecision]] = defaultdict(list)
    for row in decisions:
        region_ids = [str(value) for value in row.alignment.get("region_ids", [])]
        missing = [region_id for region_id in region_ids if region_id not in locations]
        if missing:
            row.decision = "IGNORE"
            row.reason = "alignment_region_missing_from_tokens"
            continue
        for region_id in region_ids:
            by_region[region_id].append(row)
    for rows in by_region.values():
        unique_fields = {str(row.alignment.get("field_path", "")) for row in rows}
        if len(unique_fields) <= 1:
            continue
        for row in rows:
            row.decision = "IGNORE"
            row.reason = "region_shared_by_multiple_fields"


def _clean_document(
    source: Path,
    output: Path,
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = copy.deepcopy(_load_object(path))
    raw_pages = payload.get("pages")
    raw_alignments = payload.get("alignments")
    raw_unmatched = payload.get("unmatched_fields", [])
    if not isinstance(raw_pages, list) or not all(isinstance(page, dict) for page in raw_pages):
        raise ValueError(f"layout annotation has invalid pages: {path}")
    if not isinstance(raw_alignments, list) or not all(
        isinstance(row, dict) for row in raw_alignments
    ):
        raise ValueError(f"layout annotation has invalid alignments: {path}")
    if not isinstance(raw_unmatched, list) or not all(
        isinstance(row, dict) for row in raw_unmatched
    ):
        raise ValueError(f"layout annotation has invalid unmatched_fields: {path}")

    pages: list[dict[str, Any]] = raw_pages
    locations = _page_region_map(pages)
    decisions = [_classify_alignment(row) for row in raw_alignments]
    _resolve_conflicts(decisions, locations)

    for page in pages:
        token_count = len(page["tokens"])
        page["labels"] = ["O"] * token_count
        page["ignore_mask"] = [False] * token_count
        page["image_path"] = _relative_source_image(source, output, page.get("image_path"))
        regions = page.get("regions")
        if isinstance(regions, list):
            for region in regions:
                if isinstance(region, dict):
                    region["label"] = "O"
                    region["ignore"] = False

    review_rows: list[dict[str, Any]] = []
    for row in decisions:
        alignment = row.alignment
        alignment["cleaning_decision"] = row.decision
        alignment["cleaning_reason"] = row.reason
        alignment["clean_training_eligible"] = row.decision == "KEEP"
        region_ids = [str(value) for value in alignment.get("region_ids", [])]
        if row.decision == "KEEP":
            label = str(alignment["label"])
            for region_position, region_id in enumerate(region_ids):
                page_index, token_index = locations[region_id]
                clean_label = f"{'B' if region_position == 0 else 'I'}-{label}"
                pages[page_index]["labels"][token_index] = clean_label
                regions = pages[page_index].get("regions")
                if isinstance(regions, list) and token_index < len(regions):
                    region = regions[token_index]
                    if isinstance(region, dict):
                        region["label"] = clean_label
            continue
        for region_id in region_ids:
            location = locations.get(region_id)
            if location is None:
                continue
            page_index, token_index = location
            pages[page_index]["ignore_mask"][token_index] = True
            regions = pages[page_index].get("regions")
            if isinstance(regions, list) and token_index < len(regions):
                region = regions[token_index]
                if isinstance(region, dict):
                    region["ignore"] = True
        if row.decision == "REVIEW":
            review_rows.append(
                {
                    "document_id": payload.get("document_id", path.stem),
                    **copy.deepcopy(alignment),
                }
            )

    for row in raw_unmatched:
        row["cleaning_decision"] = "IGNORE"
        row["cleaning_reason"] = "unmatched_no_region"
        row["clean_training_eligible"] = False
    payload["annotation_kind"] = "cleaned_pseudo_layout_gt"
    payload["cleaning_profile"] = "conservative"
    payload["cleaned_at"] = datetime.now(timezone.utc).isoformat()
    return payload, review_rows


def _decision_report(
    source: Path,
    output: Path,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    by_field: dict[str, Counter[str]] = defaultdict(Counter)
    token_count = 0
    ignored_token_count = 0
    for document in documents:
        for page in document["pages"]:
            token_count += len(page["tokens"])
            ignored_token_count += sum(page["ignore_mask"])
        for row in [*document.get("alignments", []), *document.get("unmatched_fields", [])]:
            decision = str(row["cleaning_decision"])
            decisions[decision] += 1
            reasons[str(row["cleaning_reason"])] += 1
            by_field[str(row["label"])][decision] += 1
    total_fields = sum(decisions.values())
    return {
        "schema_version": "cleaned-pseudo-layout-gt-report-v1",
        "source_layout_gt": str(source),
        "output": str(output),
        "profile": {
            "name": "conservative",
            "safe_methods": sorted(CONSERVATIVE_SAFE_METHODS),
            "minimum_match_confidence": 0.98,
            "minimum_recognition_confidence": 0.80,
            "minimum_detection_confidence": 0.80,
            "maximum_kept_boxes": 1,
        },
        "summary": {
            "documents": len(documents),
            "total_fields": total_fields,
            "keep_fields": decisions["KEEP"],
            "review_fields": decisions["REVIEW"],
            "ignore_fields": decisions["IGNORE"],
            "clean_training_coverage_percent": (
                decisions["KEEP"] / total_fields * 100 if total_fields else 0.0
            ),
            "tokens": token_count,
            "ignored_tokens": ignored_token_count,
            "supervised_tokens": token_count - ignored_token_count,
        },
        "decisions_by_reason": dict(sorted(reasons.items())),
        "decisions_by_field": {
            label: {
                "total_fields": sum(counts.values()),
                "keep_fields": counts["KEEP"],
                "review_fields": counts["REVIEW"],
                "ignore_fields": counts["IGNORE"],
            }
            for label, counts in sorted(by_field.items())
        },
    }


def clean_layout_ground_truth(request: LayoutGTCleanRequest) -> Path:
    """Create a conservative training dataset while preserving the source bytes."""
    source, output = _prepare_paths(request)
    source_manifest = _load_object(source / "manifest.json")
    paths = sorted((source / "layout").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no pseudo-layout annotations found under {source / 'layout'}")

    LOGGER.info(
        "Starting pseudo-layout cleaning: documents=%d profile=%s",
        len(paths),
        request.profile,
    )
    documents: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for position, path in enumerate(paths, start=1):
        cleaned, document_review_rows = _clean_document(source, output, path)
        _write_object(output / "layout" / path.name, cleaned)
        documents.append(cleaned)
        review_rows.extend(document_review_rows)
        LOGGER.info(
            "[%d/%d] Cleaned %s: keep=%d review=%d ignore=%d",
            position,
            len(paths),
            path.name,
            sum(row["cleaning_decision"] == "KEEP" for row in cleaned["alignments"]),
            sum(row["cleaning_decision"] == "REVIEW" for row in cleaned["alignments"]),
            sum(
                row["cleaning_decision"] == "IGNORE"
                for row in [*cleaned["alignments"], *cleaned.get("unmatched_fields", [])]
            ),
        )

    report = _decision_report(source, output, documents)
    _write_object(output / "cleaning_report.json", report)
    _write_object(
        output / "review_queue.json",
        {
            "schema_version": "layout-review-queue-v1",
            "profile": request.profile,
            "review_count": len(review_rows),
            "items": review_rows,
        },
    )
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "schema_version": "cleaned-pseudo-layout-gt-v1",
            "annotation_kind": "cleaned_pseudo_layout_gt",
            "source_layout_gt": str(source),
            "cleaning_profile": request.profile,
            "cleaned_at": datetime.now(timezone.utc).isoformat(),
            "layout_annotation_count": len(documents),
            "cleaning_summary": report["summary"],
        }
    )
    _write_object(output / "manifest.json", manifest)

    validation = validate_ground_truth(output)
    if not validation.is_valid:
        raise ValueError("clean layout validation failed: " + "; ".join(validation.errors))
    summary = report["summary"]
    LOGGER.info(
        "Pseudo-layout cleaning completed: keep=%d review=%d ignore=%d/%d "
        "clean_coverage=%.2f%% output=%s",
        summary["keep_fields"],
        summary["review_fields"],
        summary["ignore_fields"],
        summary["total_fields"],
        summary["clean_training_coverage_percent"],
        output,
    )
    return output

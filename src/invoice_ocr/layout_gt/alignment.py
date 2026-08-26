"""Deterministic OCR-to-field alignment for pseudo LayoutLM supervision."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher
from statistics import median
from typing import Any

from invoice_ocr.contracts import BoundingBox
from invoice_ocr.postprocessing.numbers import parse_vietnamese_number

PSEUDO_ANNOTATION_SOURCE = "pretrained_ocr_alignment"

HEADER_FIELDS = {
    "invoice_number": ("INVOICE_NUMBER", "identifier"),
    "invoice_serial": ("INVOICE_SERIAL", "identifier"),
    "invoice_date": ("INVOICE_DATE", "date"),
    "payment_method": ("PAYMENT_METHOD", "text"),
    "invoice_lookup_code": ("INVOICE_LOOKUP_CODE", "identifier"),
    "contract_reference": ("CONTRACT_REFERENCE", "identifier"),
}
SUPPLIER_FIELDS = {
    "supplier_name": ("SUPPLIER_NAME", "text"),
    "tax_code": ("SELLER_TAX_CODE", "identifier"),
    "address": ("SELLER_ADDRESS", "text"),
    "phone": ("SELLER_PHONE", "identifier"),
}
BUYER_FIELDS = {
    "buyer_contact": ("BUYER_CONTACT", "text"),
    "buyer_organization": ("BUYER_ORGANIZATION", "text"),
    "tax_code": ("BUYER_TAX_CODE", "identifier"),
    "budgetary_unit_code": ("BUYER_BUDGETARY_UNIT_CODE", "identifier"),
    "address": ("BUYER_ADDRESS", "text"),
}
TOTAL_FIELDS = {
    "subtotal_excluding_vat": ("SUBTOTAL", "money"),
    "vat_rate_percent": ("TOTAL_VAT_RATE", "number"),
    "vat_total": ("VAT_TOTAL", "money"),
    "grand_total": ("GRAND_TOTAL", "money"),
    "amount_in_words": ("AMOUNT_IN_WORDS", "text"),
}
ITEM_FIELDS = {
    "line_number": ("LINE_NUMBER", "number"),
    "raw_description": ("RAW_DESCRIPTION", "text"),
    "medicine_name": ("MEDICINE_NAME", "text"),
    "strength": ("STRENGTH", "text"),
    "manufacturer": ("MANUFACTURER", "text"),
    "country_of_manufacture": ("COUNTRY_OF_MANUFACTURE", "text"),
    "bid_package_name": ("ITEM_BID_PACKAGE", "text"),
    "contract_reference": ("ITEM_CONTRACT_REFERENCE", "identifier"),
    "lot_number": ("LOT_NUMBER", "identifier"),
    "expiry_date": ("EXPIRY_DATE", "date"),
    "unit": ("UNIT", "text"),
    "quantity": ("QUANTITY", "number"),
    "unit_price": ("UNIT_PRICE", "money"),
    "line_amount": ("LINE_AMOUNT", "money"),
    "vat_rate_percent": ("ITEM_VAT_RATE", "number"),
    "vat_amount": ("VAT_AMOUNT", "money"),
}
WORKFLOW_FIELDS = {
    "status": ("STATUS", "text"),
    "invoice_type": ("INVOICE_TYPE", "text"),
    "bid_package": ("BID_PACKAGE", "text"),
    "delivery_unit": ("DELIVERY_UNIT", "text"),
    "receiver_name": ("RECEIVER_NAME", "text"),
}
SIMPLE_FIELD_LABELS = {
    **HEADER_FIELDS,
    **SUPPLIER_FIELDS,
    **BUYER_FIELDS,
    **TOTAL_FIELDS,
    **ITEM_FIELDS,
    **WORKFLOW_FIELDS,
}

_SMART_PUNCTUATION = str.maketrans(
    {
        "\u00a0": " ",
        "“": '"',
        "”": '"',
        "„": '"',
        "\u2019": "'",
        "\u2018": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\uff1a": ":",
        "\uff0c": ",",
        "\uff0e": ".",
        "\uff0f": "/",
    }
)
_WHITESPACE = re.compile(r"\s+")
_TEXT_PUNCTUATION = re.compile(r"[\s:;|,_]+")
_IDENTIFIER_PUNCTUATION = re.compile(r"[^0-9a-zA-ZÀ-ỹ]+", re.UNICODE)
_DATE_NUMERIC = re.compile(r"(?<!\d)(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})(?!\d)")
_DATE_VIETNAMESE = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE,
)


_FIELD_VALUE_PREFIXES = {
    "INVOICE_NUMBER": ("so hoa don", "so"),
    "INVOICE_SERIAL": ("ky hieu hoa don", "ky hieu"),
    "PAYMENT_METHOD": ("hinh thuc thanh toan",),
    "SELLER_TAX_CODE": ("ma so thue",),
    "BUYER_TAX_CODE": ("ma so thue",),
    "SUBTOTAL": ("tong tien chua thue", "tong tien hang", "cong tien hang"),
    "TOTAL_VAT_RATE": ("thue suat gtgt", "thue suat"),
    "VAT_TOTAL": ("tien thue gtgt", "thue gtgt"),
    "GRAND_TOTAL": ("tong cong tien thanh toan", "tong tien thanh toan"),
    "AMOUNT_IN_WORDS": ("so tien viet bang chu",),
    "EXPIRY_DATE": ("han su dung", "han dung"),
}
_DATE_VALUE_TEXT = re.compile(
    r"^(?:\d{1,4}[./-]\d{1,2}[./-]\d{1,4}|"
    r"ngay\s+\d{1,2}\s+thang\s+\d{1,2}\s+nam\s+\d{4})$",
    re.IGNORECASE,
)
_NUMERIC_VALUE_TEXT = re.compile(
    r"^[+-]?\s*\d+(?:[.\s]\d+)*\s*(?:%|vnd|d|dong)?$",
    re.IGNORECASE,
)
_NUMERIC_SUFFIX = re.compile(r"\s*(?:%|vnd|d|dong)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class GroundTruthField:
    field_path: str
    label: str
    gt_value: str
    normalization_kind: str
    page_index: int | None
    invoice_index: int
    item_index: int | None = None
    line_number: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_path": self.field_path,
            "label": self.label,
            "gt_value": self.gt_value,
            "page": self.page_index + 1 if self.page_index is not None else None,
            "invoice_index": self.invoice_index,
            "item_index": self.item_index,
            "line_number": self.line_number,
        }


@dataclass(frozen=True)
class OCRRegion:
    region_id: str
    page_index: int
    text: str
    bbox: BoundingBox
    polygon: list[list[float]]
    detection_confidence: float
    recognition_confidence: float

    @property
    def center_x(self) -> float:
        return (self.bbox.x_min + self.bbox.x_max) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox.y_min + self.bbox.y_max) / 2


@dataclass(frozen=True)
class MatchCandidate:
    regions: tuple[OCRRegion, ...]
    method: str
    confidence: float
    inherently_ambiguous: bool = False

    @property
    def text(self) -> str:
        return " ".join(region.text.strip() for region in self.regions if region.text.strip())

    @property
    def center_x(self) -> float:
        return sum(region.center_x for region in self.regions) / len(self.regions)

    @property
    def center_y(self) -> float:
        return sum(region.center_y for region in self.regions) / len(self.regions)

    @property
    def mean_height(self) -> float:
        return sum(region.bbox.y_max - region.bbox.y_min for region in self.regions) / len(
            self.regions
        )


@dataclass(frozen=True)
class FieldAlignment:
    field: GroundTruthField
    candidate: MatchCandidate
    match_method: str
    match_confidence: float
    ambiguous: bool
    duplicate_candidates: bool
    candidate_count: int

    @property
    def training_eligible(self) -> bool:
        return not self.ambiguous

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_path": self.field.field_path,
            "label": self.field.label,
            "gt_value": self.field.gt_value,
            "ocr_text": self.candidate.text,
            "region_ids": [region.region_id for region in self.candidate.regions],
            "boxes": [
                [
                    region.bbox.x_min,
                    region.bbox.y_min,
                    region.bbox.x_max,
                    region.bbox.y_max,
                ]
                for region in self.candidate.regions
            ],
            "polygons": [region.polygon for region in self.candidate.regions],
            "page": self.candidate.regions[0].page_index + 1,
            "detection_confidence": min(
                region.detection_confidence for region in self.candidate.regions
            ),
            "recognition_confidence": min(
                region.recognition_confidence for region in self.candidate.regions
            ),
            "match_method": self.match_method,
            "match_confidence": self.match_confidence,
            "ambiguous": self.ambiguous,
            "duplicate_candidates": self.duplicate_candidates,
            "candidate_count": self.candidate_count,
            "training_eligible": self.training_eligible,
            "source": PSEUDO_ANNOTATION_SOURCE,
            "invoice_index": self.field.invoice_index,
            "item_index": self.field.item_index,
            "line_number": self.field.line_number,
        }


@dataclass
class AlignmentResult:
    fields: list[GroundTruthField]
    matches: list[FieldAlignment]
    unmatched: list[GroundTruthField]
    region_labels: dict[str, str] = field(default_factory=dict)


def _text_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text if text else None


def _page_index(invoice: dict[str, Any], invoice_index: int) -> int:
    raw = invoice.get("page_number", invoice_index + 1)
    try:
        return max(0, int(raw) - 1)
    except (TypeError, ValueError):
        return invoice_index


def _append_mapping_fields(
    result: list[GroundTruthField],
    section: dict[str, Any],
    mapping: dict[str, tuple[str, str]],
    path_prefix: str,
    page_index: int,
    invoice_index: int,
    item_index: int | None = None,
    line_number: int | None = None,
) -> None:
    for name, (label, kind) in mapping.items():
        value = _text_value(section.get(name))
        if value is None:
            continue
        result.append(
            GroundTruthField(
                field_path=f"{path_prefix}.{name}",
                label=label,
                gt_value=value,
                normalization_kind=kind,
                page_index=page_index,
                invoice_index=invoice_index,
                item_index=item_index,
                line_number=line_number,
            )
        )


def flatten_canonical_ground_truth(payload: dict[str, Any]) -> list[GroundTruthField]:
    """Flatten visible canonical values while excluding derived validation metadata."""
    if isinstance(payload.get("field"), str) and "text" in payload:
        name = str(payload["field"])
        simple_mapping = SIMPLE_FIELD_LABELS.get(name)
        value = _text_value(payload.get("text"))
        if simple_mapping is None or value is None:
            return []
        raw_page = payload.get("page", 1)
        try:
            page_index = max(0, int(raw_page) - 1)
        except (TypeError, ValueError):
            page_index = 0
        return [
            GroundTruthField(
                field_path=name,
                label=simple_mapping[0],
                gt_value=value,
                normalization_kind=simple_mapping[1],
                page_index=page_index,
                invoice_index=0,
            )
        ]
    invoices = payload.get("invoices")
    if not isinstance(invoices, list):
        invoices = (
            [payload] if any(key in payload for key in ("invoice", "supplier", "items")) else []
        )
    result: list[GroundTruthField] = []
    for invoice_index, raw_invoice in enumerate(invoices):
        if not isinstance(raw_invoice, dict):
            continue
        page_index = _page_index(raw_invoice, invoice_index)
        base = f"invoices[{invoice_index}]"
        for section_name, mapping in (
            ("invoice", HEADER_FIELDS),
            ("supplier", SUPPLIER_FIELDS),
            ("buyer", BUYER_FIELDS),
            ("totals", TOTAL_FIELDS),
        ):
            section = raw_invoice.get(section_name)
            if isinstance(section, dict):
                _append_mapping_fields(
                    result,
                    section,
                    mapping,
                    f"{base}.{section_name}",
                    page_index,
                    invoice_index,
                )
        workflow = raw_invoice.get("workflow_fields")
        if isinstance(workflow, dict):
            for name, (label, kind) in WORKFLOW_FIELDS.items():
                entry = workflow.get(name)
                if not isinstance(entry, dict) or entry.get("source") != "invoice":
                    continue
                value = _text_value(entry.get("value"))
                if value is not None:
                    result.append(
                        GroundTruthField(
                            field_path=f"{base}.workflow_fields.{name}.value",
                            label=label,
                            gt_value=value,
                            normalization_kind=kind,
                            page_index=page_index,
                            invoice_index=invoice_index,
                        )
                    )
        items = raw_invoice.get("items")
        if not isinstance(items, list):
            continue
        for item_index, raw_item in enumerate(items):
            if not isinstance(raw_item, dict):
                continue
            raw_line_number = raw_item.get("line_number", item_index + 1)
            try:
                line_number = int(raw_line_number)
            except (TypeError, ValueError):
                line_number = item_index + 1
            _append_mapping_fields(
                result,
                raw_item,
                ITEM_FIELDS,
                f"{base}.items[{item_index}]",
                page_index,
                invoice_index,
                item_index,
                line_number,
            )
    return result


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(_SMART_PUNCTUATION).casefold()
    normalized = _TEXT_PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip(" .-'\"")


def _fold_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", normalize_text(value))
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return without_marks.replace("\u0111", "d")


def _anchored_value(target: GroundTruthField, ocr: str) -> str | None:
    folded_ocr = _fold_text(ocr)
    for prefix in _FIELD_VALUE_PREFIXES.get(target.label, ()):
        marker = f"{prefix} "
        if folded_ocr.startswith(marker):
            value = folded_ocr[len(marker) :].strip()
            return value or None
    return None


def _numeric_candidate_text(target: GroundTruthField, ocr: str) -> str | None:
    anchored = _anchored_value(target, ocr)
    candidate = anchored if anchored is not None else _fold_text(ocr)
    if not _NUMERIC_VALUE_TEXT.fullmatch(candidate):
        return None
    return _NUMERIC_SUFFIX.sub("", candidate).strip()


def _date_candidate_text(target: GroundTruthField, ocr: str) -> str | None:
    folded = _fold_text(ocr)
    if _DATE_VALUE_TEXT.fullmatch(folded):
        return ocr
    anchored = _anchored_value(target, ocr)
    if anchored is not None and _DATE_VALUE_TEXT.fullmatch(anchored):
        return anchored
    return None


def _normalize_identifier(value: str) -> str:
    return _IDENTIFIER_PUNCTUATION.sub("", normalize_text(value))


def _normalize_number(value: str) -> Decimal | None:
    parsed = parse_vietnamese_number(value)
    return parsed.normalize() if parsed is not None else None


def _normalize_date(value: str) -> str | None:
    text = unicodedata.normalize("NFKC", value).translate(_SMART_PUNCTUATION)
    match = _DATE_VIETNAMESE.search(text)
    if match:
        day, month, year = (int(part) for part in match.groups())
    else:
        match = _DATE_NUMERIC.search(text)
        if not match:
            return None
        first, second, third = match.groups()
        if len(first) == 4:
            year, month, day = int(first), int(second), int(third)
        elif len(third) == 4:
            day, month, year = int(first), int(second), int(third)
        else:
            return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_layout_bbox(box: BoundingBox, width: int, height: int) -> list[int]:
    """Normalize a pixel box to LayoutLM's inclusive integer 0..1000 coordinate space."""
    normalized = box.normalize(width, height)
    return [
        min(1000, max(0, round(normalized.x_min))),
        min(1000, max(0, round(normalized.y_min))),
        min(1000, max(0, round(normalized.x_max))),
        min(1000, max(0, round(normalized.y_max))),
    ]


def _candidate_match(
    target: GroundTruthField,
    regions: tuple[OCRRegion, ...],
) -> MatchCandidate | None:
    ocr = " ".join(region.text.strip() for region in regions if region.text.strip())
    if not ocr:
        return None
    gt = target.gt_value.strip()
    multi = len(regions) > 1
    prefix = "multi_box_" if multi else ""
    if ocr == gt:
        return MatchCandidate(regions, f"{prefix}exact", 1.0)
    if _WHITESPACE.sub(" ", ocr).strip() == _WHITESPACE.sub(" ", gt).strip():
        return MatchCandidate(regions, f"{prefix}whitespace_normalized", 0.995)
    normalized_ocr = normalize_text(ocr)
    normalized_gt = normalize_text(gt)
    anchored_ocr = _anchored_value(target, ocr)
    if anchored_ocr is not None and anchored_ocr == _fold_text(gt):
        return MatchCandidate(regions, f"{prefix}anchored_exact_normalized", 0.985)
    if normalized_ocr and normalized_ocr == normalized_gt:
        return MatchCandidate(regions, f"{prefix}exact_normalized", 0.99)
    if target.normalization_kind == "identifier":
        identifier_ocr = _normalize_identifier(ocr)
        identifier_gt = _normalize_identifier(gt)
        if identifier_ocr and identifier_ocr == identifier_gt:
            return MatchCandidate(regions, f"{prefix}identifier_normalized", 0.98)
    if target.normalization_kind in {"number", "money"}:
        number_source = _numeric_candidate_text(target, ocr)
        number_ocr = _normalize_number(number_source) if number_source is not None else None
        number_gt = _normalize_number(gt)
        if number_ocr is not None and number_ocr == number_gt:
            method = (
                "money_normalized" if target.normalization_kind == "money" else "number_normalized"
            )
            return MatchCandidate(regions, f"{prefix}{method}", 0.98)
    if target.normalization_kind == "date":
        date_source = _date_candidate_text(target, ocr)
        date_ocr = _normalize_date(date_source) if date_source is not None else None
        date_gt = _normalize_date(gt)
        if date_ocr is not None and date_ocr == date_gt:
            return MatchCandidate(regions, f"{prefix}date_normalized", 0.98)
    if len(normalized_gt) >= 3 and (
        normalized_gt in normalized_ocr or normalized_ocr in normalized_gt
    ):
        shorter = min(len(normalized_gt), len(normalized_ocr))
        longer = max(len(normalized_gt), len(normalized_ocr))
        ratio = shorter / longer if longer else 0.0
        if ratio >= 0.55:
            return MatchCandidate(
                regions,
                f"{prefix}value_substring",
                min(0.89, 0.72 + ratio * 0.15),
                inherently_ambiguous=True,
            )
    if min(len(normalized_gt), len(normalized_ocr)) >= 4:
        similarity = SequenceMatcher(None, normalized_gt, normalized_ocr).ratio()
        if similarity >= 0.84:
            return MatchCandidate(
                regions,
                f"{prefix}fuzzy",
                similarity,
                inherently_ambiguous=True,
            )
    return None


def _prune_redundant_spans(
    candidates: list[MatchCandidate],
) -> list[MatchCandidate]:
    region_sets = [{region.region_id for region in candidate.regions} for candidate in candidates]
    result: list[MatchCandidate] = []
    for index, candidate in enumerate(candidates):
        redundant = any(
            other_index != index
            and region_sets[other_index] < region_sets[index]
            and other.confidence >= candidate.confidence
            for other_index, other in enumerate(candidates)
        )
        if not redundant:
            result.append(candidate)
    return result


def _candidate_regions_for_field(
    target: GroundTruthField,
    regions: list[OCRRegion],
    max_boxes: int,
) -> list[MatchCandidate]:
    pages = sorted({region.page_index for region in regions})
    if target.page_index in pages:
        allowed_pages = {target.page_index}
    elif len(pages) == 1:
        allowed_pages = set(pages)
    else:
        allowed_pages = set(pages)
    matches: list[MatchCandidate] = []
    for page_index in allowed_pages:
        page_regions = sorted(
            (region for region in regions if region.page_index == page_index),
            key=lambda region: (region.bbox.y_min, region.bbox.x_min, region.region_id),
        )
        for start in range(len(page_regions)):
            for length in range(1, min(max_boxes, len(page_regions) - start) + 1):
                span = tuple(page_regions[start : start + length])
                candidate = _candidate_match(target, span)
                if candidate is not None:
                    matches.append(candidate)
    best_by_regions: dict[tuple[str, ...], MatchCandidate] = {}
    for candidate in matches:
        key = tuple(region.region_id for region in candidate.regions)
        existing = best_by_regions.get(key)
        if existing is None or candidate.confidence > existing.confidence:
            best_by_regions[key] = candidate
    candidates = _prune_redundant_spans(list(best_by_regions.values()))
    strong = [candidate for candidate in candidates if not candidate.inherently_ambiguous]
    selected = strong if strong else candidates
    return sorted(
        selected,
        key=lambda candidate: (
            -candidate.confidence,
            len(candidate.regions),
            candidate.regions[0].page_index,
            candidate.center_y,
            candidate.center_x,
        ),
    )


def _item_key(field: GroundTruthField) -> tuple[int, int] | None:
    if field.item_index is None:
        return None
    return field.invoice_index, field.item_index


def _choose_item_candidate(
    field: GroundTruthField,
    candidates: list[MatchCandidate],
    all_fields: list[GroundTruthField],
    all_candidates: dict[str, list[MatchCandidate]],
    row_centers: dict[tuple[int, int], list[float]],
    column_centers: dict[str, list[float]],
) -> tuple[MatchCandidate, str, float] | None:
    item_key = _item_key(field)
    if item_key is None:
        return None
    expected_row = median(row_centers[item_key]) if row_centers.get(item_key) else None
    expected_column = (
        median(column_centers[field.label]) if column_centers.get(field.label) else None
    )
    if expected_row is not None or expected_column is not None:
        scored: list[tuple[float, MatchCandidate]] = []
        for candidate in candidates:
            score = 0.0
            if expected_row is not None:
                score += abs(candidate.center_y - expected_row) / max(candidate.mean_height, 1.0)
            if expected_column is not None:
                width = max(
                    candidate.regions[-1].bbox.x_max - candidate.regions[0].bbox.x_min,
                    1.0,
                )
                score += abs(candidate.center_x - expected_column) / width
            scored.append((score, candidate))
        scored.sort(key=lambda pair: (pair[0], -pair[1].confidence, pair[1].center_y))
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else float("inf")
        if best_score <= 4.0 and second_score - best_score >= 0.5:
            confidence = min(best.confidence, max(0.85, 0.98 - best_score * 0.03))
            return best, "item_row_column_geometry", confidence
    peers = [
        peer
        for peer in all_fields
        if peer.invoice_index == field.invoice_index
        and peer.item_index is not None
        and peer.label == field.label
        and normalize_text(peer.gt_value) == normalize_text(field.gt_value)
        and peer.page_index == field.page_index
    ]
    unique_candidates: dict[tuple[str, ...], MatchCandidate] = {}
    for peer in peers:
        for candidate in all_candidates.get(peer.field_path, []):
            key = tuple(region.region_id for region in candidate.regions)
            unique_candidates.setdefault(key, candidate)
    ordered_candidates = sorted(
        unique_candidates.values(), key=lambda candidate: (candidate.center_y, candidate.center_x)
    )
    ordered_peers = sorted(
        peers,
        key=lambda peer: peer.item_index if peer.item_index is not None else -1,
    )
    if len(ordered_candidates) == len(ordered_peers) and len(ordered_peers) > 1:
        peer_index = ordered_peers.index(field)
        selected = ordered_candidates[peer_index]
        if any(
            tuple(region.region_id for region in selected.regions)
            == tuple(region.region_id for region in candidate.regions)
            for candidate in candidates
        ):
            return selected, "item_order", min(0.94, selected.confidence)
    return None


def align_ground_truth_fields(
    fields: list[GroundTruthField],
    regions: list[OCRRegion],
    *,
    max_boxes: int = 12,
) -> AlignmentResult:
    """Align fields conservatively; uncertain candidates remain visible but are not trained."""
    if max_boxes <= 0:
        raise ValueError("max_boxes must be positive")
    candidates_by_field = {
        target.field_path: _candidate_regions_for_field(target, regions, max_boxes)
        for target in fields
    }
    anchor_priority = {
        "LINE_NUMBER": 0,
        "MEDICINE_NAME": 1,
        "RAW_DESCRIPTION": 2,
        "LOT_NUMBER": 3,
    }
    ordered_fields = sorted(
        fields,
        key=lambda target: (
            target.item_index is not None,
            target.invoice_index,
            target.item_index if target.item_index is not None else -1,
            anchor_priority.get(target.label, 10),
            target.field_path,
        ),
    )
    used_region_ids: set[str] = set()
    row_centers: dict[tuple[int, int], list[float]] = {}
    column_centers: dict[str, list[float]] = {}
    matches_by_path: dict[str, FieldAlignment] = {}
    unmatched_paths: set[str] = set()
    for target in ordered_fields:
        all_candidates = candidates_by_field[target.field_path]
        candidates = [
            candidate
            for candidate in all_candidates
            if not any(region.region_id in used_region_ids for region in candidate.regions)
        ]
        if not candidates:
            unmatched_paths.add(target.field_path)
            continue
        duplicate = len(all_candidates) > 1
        selected: MatchCandidate
        method: str
        confidence: float
        ambiguous: bool
        if len(candidates) == 1:
            selected = candidates[0]
            method = selected.method
            confidence = selected.confidence
            ambiguous = selected.inherently_ambiguous
        else:
            item_choice = _choose_item_candidate(
                target,
                candidates,
                fields,
                candidates_by_field,
                row_centers,
                column_centers,
            )
            if item_choice is not None:
                selected, method, confidence = item_choice
                ambiguous = selected.inherently_ambiguous
            else:
                selected = candidates[0]
                method = selected.method
                confidence = min(selected.confidence, 0.75)
                ambiguous = True
        alignment = FieldAlignment(
            field=target,
            candidate=selected,
            match_method=method,
            match_confidence=confidence,
            ambiguous=ambiguous,
            duplicate_candidates=duplicate,
            candidate_count=len(all_candidates),
        )
        matches_by_path[target.field_path] = alignment
        if ambiguous:
            continue
        for region in selected.regions:
            used_region_ids.add(region.region_id)
        key = _item_key(target)
        if key is not None:
            row_centers.setdefault(key, []).append(selected.center_y)
            column_centers.setdefault(target.label, []).append(selected.center_x)
    matches = [
        matches_by_path[field.field_path] for field in fields if field.field_path in matches_by_path
    ]
    unmatched = [field for field in fields if field.field_path in unmatched_paths]
    region_labels: dict[str, str] = {}
    for match in matches:
        if not match.training_eligible:
            continue
        for index, region in enumerate(match.candidate.regions):
            prefix = "B" if index == 0 else "I"
            region_labels[region.region_id] = f"{prefix}-{match.field.label}"
    return AlignmentResult(
        fields=fields,
        matches=matches,
        unmatched=unmatched,
        region_labels=region_labels,
    )

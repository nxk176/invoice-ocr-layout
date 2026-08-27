"""Canonical JSON comparison metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from invoice_ocr.evaluation.recognition import edit_distance


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_json(nested, child))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            flattened.update(flatten_json(nested, f"{prefix}[{index}]"))
    else:
        flattened[prefix] = value
    return flattened


def normalize_scalar(value: Any) -> Any:
    return " ".join(value.casefold().split()) if isinstance(value, str) else value


def canonical_payload_view(value: Any) -> Any:
    """Ignore review metadata while retaining every canonical field and list position."""
    if isinstance(value, Mapping):
        return {
            str(key): canonical_payload_view(nested)
            for key, nested in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [canonical_payload_view(nested) for nested in value]
    return value


def _canonical_fields(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = canonical_payload_view(payload)
    if not isinstance(canonical, dict):
        raise ValueError("canonical invoice payload must be a JSON object")
    return {
        path: value
        for path, value in flatten_json(canonical).items()
        if _is_populated_business_field(path, value)
    }


def _is_populated_business_field(path: str, value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return False
    if path in {"document_type", "invoice_count"} or path.endswith(".page_number"):
        return False
    if ".validation." in path:
        return False
    return ".workflow_fields." not in path or path.endswith(".value")


def _value_token(value: Any, *, normalized: bool) -> str:
    selected = normalize_scalar(value) if normalized else value
    return repr(selected)


def field_level_counts(
    predicted: dict[str, Any],
    expected: dict[str, Any],
    *,
    normalized: bool = False,
) -> dict[str, int]:
    """Count exact path-value pairs; a wrong value is one FP and one FN."""
    predicted_fields = _canonical_fields(predicted)
    expected_fields = _canonical_fields(expected)
    predicted_pairs = {
        (path, _value_token(value, normalized=normalized))
        for path, value in predicted_fields.items()
    }
    expected_pairs = {
        (path, _value_token(value, normalized=normalized))
        for path, value in expected_fields.items()
    }
    return {
        "true_positives": len(predicted_pairs & expected_pairs),
        "predicted_fields": len(predicted_pairs),
        "expected_fields": len(expected_pairs),
    }


def field_character_error_counts(
    predicted: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, int]:
    """Count micro character edits by canonical field path, including missing/extras."""
    predicted_fields = _canonical_fields(predicted)
    expected_fields = _canonical_fields(expected)
    errors = 0
    expected_characters = 0
    for path, expected_value in expected_fields.items():
        expected_text = "" if expected_value is None else str(expected_value)
        predicted_value = predicted_fields.get(path)
        predicted_text = "" if predicted_value is None else str(predicted_value)
        errors += edit_distance(list(predicted_text), list(expected_text))
        expected_characters += len(expected_text)
    for path in predicted_fields.keys() - expected_fields.keys():
        predicted_value = predicted_fields[path]
        errors += len("" if predicted_value is None else str(predicted_value))
    return {
        "character_errors": errors,
        "expected_characters": expected_characters,
    }


def precision_recall_f1_from_counts(counts: dict[str, int]) -> tuple[float, float, float]:
    """Derive micro field metrics from canonical path-value pair counts."""
    true_positives = counts["true_positives"]
    predicted_fields = counts["predicted_fields"]
    expected_fields = counts["expected_fields"]
    precision = (
        true_positives / predicted_fields
        if predicted_fields
        else (1.0 if not expected_fields else 0.0)
    )
    recall = (
        true_positives / expected_fields
        if expected_fields
        else (1.0 if not predicted_fields else 0.0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    invoices = payload.get("invoices", [])
    if not isinstance(invoices, list):
        return rows
    for invoice in invoices:
        if not isinstance(invoice, Mapping):
            continue
        items = invoice.get("items", [])
        if isinstance(items, list):
            rows.extend(dict(item) for item in items if isinstance(item, Mapping))
    return rows


def _normalized_row(row: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(key), normalize_scalar(value)) for key, value in row.items()))


def _item_metrics(predicted: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    predicted_rows = _items(predicted)
    expected_rows = _items(expected)
    predicted_counter = Counter(_normalized_row(row) for row in predicted_rows)
    expected_counter = Counter(_normalized_row(row) for row in expected_rows)
    exact_rows = sum((predicted_counter & expected_counter).values())
    precision = (
        exact_rows / len(predicted_rows) if predicted_rows else (1.0 if not expected_rows else 0.0)
    )
    recall = (
        exact_rows / len(expected_rows) if expected_rows else (1.0 if not predicted_rows else 0.0)
    )
    row_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    remaining = list(predicted_rows)
    correct_fields = 0
    total_fields = sum(len(row) for row in expected_rows)
    for expected_row in expected_rows:
        if not remaining:
            continue
        scored = [
            (
                sum(
                    normalize_scalar(candidate.get(key)) == normalize_scalar(expected_value)
                    for key, expected_value in expected_row.items()
                ),
                index,
            )
            for index, candidate in enumerate(remaining)
        ]
        best_score, best_index = max(scored, key=lambda pair: (pair[0], -pair[1]))
        correct_fields += best_score
        remaining.pop(best_index)
    return {
        "medicine_row_matching": row_f1,
        "item_row_precision": precision,
        "item_row_recall": recall,
        "item_row_f1": row_f1,
        "item_field_accuracy": (
            correct_fields / total_fields if total_fields else (1.0 if not predicted_rows else 0.0)
        ),
    }


def final_json_metrics(predicted: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    canonical_predicted = canonical_payload_view(predicted)
    canonical_expected = canonical_payload_view(expected)
    if not isinstance(canonical_predicted, dict) or not isinstance(canonical_expected, dict):
        raise ValueError("canonical invoice payloads must be JSON objects")
    predicted_fields = flatten_json(canonical_predicted)
    expected_fields = flatten_json(canonical_expected)
    keys = sorted(expected_fields)
    exact = sum(predicted_fields.get(key) == expected_fields[key] for key in keys)
    normalized = sum(
        normalize_scalar(predicted_fields.get(key)) == normalize_scalar(expected_fields[key])
        for key in keys
    )
    exact_counts = field_level_counts(predicted, expected)
    normalized_counts = field_level_counts(predicted, expected, normalized=True)
    precision, recall, f1 = precision_recall_f1_from_counts(exact_counts)
    normalized_precision, normalized_recall, normalized_f1 = precision_recall_f1_from_counts(
        normalized_counts
    )
    character_counts = field_character_error_counts(predicted, expected)
    expected_characters = character_counts["expected_characters"]
    return {
        "field_exact_match": exact / len(keys) if keys else 1.0,
        "normalized_field_accuracy": normalized / len(keys) if keys else 1.0,
        "field_level_precision": precision,
        "field_level_recall": recall,
        "field_level_f1": f1,
        "normalized_field_level_precision": normalized_precision,
        "normalized_field_level_recall": normalized_recall,
        "normalized_field_level_f1": normalized_f1,
        "character_error_rate": (
            character_counts["character_errors"] / expected_characters
            if expected_characters
            else 0.0
        ),
        **_item_metrics(canonical_predicted, canonical_expected),
        "document_exact_match": 1.0 if canonical_predicted == canonical_expected else 0.0,
    }

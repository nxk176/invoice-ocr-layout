"""Canonical JSON comparison metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any


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
    return {
        "field_exact_match": exact / len(keys) if keys else 1.0,
        "normalized_field_accuracy": normalized / len(keys) if keys else 1.0,
        **_item_metrics(canonical_predicted, canonical_expected),
        "document_exact_match": 1.0 if canonical_predicted == canonical_expected else 0.0,
    }

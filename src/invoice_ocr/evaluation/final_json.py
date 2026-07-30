"""Canonical JSON comparison metrics."""

from __future__ import annotations

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


def final_json_metrics(predicted: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    predicted_fields = flatten_json(predicted)
    expected_fields = flatten_json(expected)
    keys = sorted(expected_fields)
    exact = sum(predicted_fields.get(key) == expected_fields[key] for key in keys)
    normalized = sum(
        normalize_scalar(predicted_fields.get(key)) == normalize_scalar(expected_fields[key])
        for key in keys
    )
    expected_rows = sum(len(invoice.get("items", [])) for invoice in expected.get("invoices", []))
    predicted_rows = sum(len(invoice.get("items", [])) for invoice in predicted.get("invoices", []))
    return {
        "field_exact_match": exact / len(keys) if keys else 1.0,
        "normalized_field_accuracy": normalized / len(keys) if keys else 1.0,
        "medicine_row_matching": 1.0 if predicted_rows == expected_rows else 0.0,
        "document_exact_match": 1.0 if predicted == expected else 0.0,
    }

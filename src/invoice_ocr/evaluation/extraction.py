"""Entity and canonical JSON comparison helpers."""

from invoice_ocr.evaluation.final_json import (
    final_json_metrics,
    flatten_json,
    normalize_scalar,
)

__all__ = ["final_json_metrics", "flatten_json", "normalize_scalar"]

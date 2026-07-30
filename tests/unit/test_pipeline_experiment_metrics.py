from __future__ import annotations

from invoice_ocr.experiments.evaluate_pipeline import _aggregate_final


def test_missing_final_gt_produces_na_not_zero() -> None:
    metrics, per_document, per_field = _aggregate_final(
        predictions={"synthetic-doc": {"document_type": "VAT_INVOICE_BATCH"}},
        expected_by_document={},
        skipped_count=1,
    )
    entry = metrics["final_normalized_field_accuracy"]
    assert entry["value"] is None
    assert entry["na_reason"] == "canonical final test ground truth is missing under GT/final"
    assert entry["skipped_sample_count"] == 1
    assert per_document == {}
    assert per_field == {}


def test_final_metrics_include_counts_and_per_document_values() -> None:
    expected = {
        "document_type": "VAT_INVOICE_BATCH",
        "invoice_count": 0,
        "invoices": [],
    }
    metrics, per_document, per_field = _aggregate_final(
        predictions={"synthetic-doc": expected},
        expected_by_document={"synthetic-doc": expected},
        skipped_count=0,
    )
    entry = metrics["final_document_exact_match"]
    assert entry["value"] == 1.0
    assert entry["numerator"] == 1.0
    assert entry["denominator"] == 1
    assert entry["evaluated_sample_count"] == 1
    assert per_document["synthetic-doc"]["final_document_exact_match"] == 1.0
    assert per_field

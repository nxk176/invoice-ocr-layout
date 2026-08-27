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


def test_final_field_metrics_use_micro_counts_and_penalize_missing_predictions() -> None:
    metrics, per_document, per_field = _aggregate_final(
        predictions={
            "synthetic-a": {"correct": "x", "extra": "q"},
        },
        expected_by_document={
            "synthetic-a": {"correct": "x", "missing": "y"},
            "synthetic-b": {"missing_prediction": "z"},
        },
        skipped_count=0,
    )

    precision = metrics["final_field_level_precision"]
    recall = metrics["final_field_level_recall"]
    f1 = metrics["final_field_level_f1"]
    cer = metrics["final_character_error_rate"]
    assert precision["value"] == 0.5
    assert precision["numerator"] == 1
    assert precision["denominator"] == 2
    assert recall["value"] == 1 / 3
    assert recall["numerator"] == 1
    assert recall["denominator"] == 3
    assert f1["value"] == 0.4
    assert f1["numerator"] == 2
    assert f1["denominator"] == 5
    assert cer["value"] == 1.0
    assert cer["numerator"] == 3
    assert cer["denominator"] == 3
    assert cer["lower_is_better"] is True
    assert per_document["synthetic-b"]["final_field_level_recall"] == 0.0
    assert per_field["missing_prediction"] == 0.0

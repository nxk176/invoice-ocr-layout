from __future__ import annotations

import json
from pathlib import Path

from invoice_ocr.experiments.split import create_locked_split
from invoice_ocr.layout_gt.index import build_final_ground_truth_index
from invoice_ocr.stages import evaluate_prediction_directory


def _write_object(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_t5_index_selects_425_pdf_targets_and_excludes_123_original_json(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data" / "t5"
    final = tmp_path / "GT" / "final" / "t5"
    predictions = tmp_path / "predictions"
    expected = {
        "document_type": "VAT_INVOICE_BATCH",
        "invoice_count": 0,
        "invoices": [],
    }
    for folder_index in range(123):
        folder = f"synthetic-source-{folder_index:03d}"
        _write_object(final / folder / f"{folder}.json", expected)
    for document_index in range(425):
        folder = f"synthetic-source-{document_index % 123:03d}"
        name = f"invoice-{document_index:04d}"
        source = data / folder / f"{name}.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"synthetic-pdf-{document_index}".encode())
        _write_object(final / folder / f"{name}.json", expected)
        _write_object(predictions / folder / f"{name}.json", expected)

    index_path = tmp_path / "work" / "t5_gt_index.json"
    index = build_final_ground_truth_index(
        data,
        tmp_path / "GT",
        index_path,
    )
    assert index.target_count == 425
    assert index.excluded_gt_count == 123
    assert index.gt_prefix == "t5"
    assert all(
        Path(target.source_relative_path).with_suffix(".json").as_posix()
        == target.prediction_relative_path
        for target in index.documents
    )
    assert not any(
        Path(target.gt_relative_path).stem.startswith("synthetic-source-")
        and Path(target.gt_relative_path).stem == Path(target.gt_relative_path).parent.name
        for target in index.documents
    )
    split = create_locked_split(
        data,
        tmp_path / "GT",
        tmp_path / "GT" / "splits" / "t5_split.json",
        seed=42,
    )
    assert len(split.dataset_documents) == 425
    assert (
        len(split.train_document_ids)
        + len(split.validation_document_ids)
        + len(split.test_document_ids)
        == 425
    )
    assert split.grouping_rules["target_selection"] == "canonical_final_gt_index"

    output = tmp_path / "evaluation.json"
    result = evaluate_prediction_directory(
        predictions,
        tmp_path / "GT",
        output,
        data_root=data,
    )
    final_result = result["final"]
    assert isinstance(final_result, dict)
    assert final_result["target_documents"] == 425
    assert final_result["evaluated_documents"] == 425
    assert final_result["missing_predictions"] == 0

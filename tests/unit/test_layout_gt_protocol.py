from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from invoice_ocr.cli import build_parser
from invoice_ocr.contracts import InvoiceBatch, InvoiceDocument
from invoice_ocr.experiments.evaluate_pipeline import (
    PipelineEvaluationRequest,
    evaluate_pipeline,
)
from invoice_ocr.experiments.split import create_locked_split
from invoice_ocr.experiments.train_model import (
    LockedTrainingRequest,
    TrainingBackendResult,
    train_model_locked,
)
from invoice_ocr.experiments.training_modes import CheckpointCandidate
from invoice_ocr.io.paths import discover_documents
from invoice_ocr.pipeline import PipelineSelection


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_layout_gt_cli_accepts_independent_ocr_devices() -> None:
    args = build_parser().parse_args(
        [
            "build-layout-gt",
            "--input",
            "data/t5",
            "--gt",
            "GT",
            "--detector",
            "paddleocr",
            "--recognizer",
            "vietocr",
            "--detector-device",
            "cpu",
            "--recognizer-device",
            "cuda",
            "--output",
            "work/layout_gt/t5",
        ]
    )

    assert args.device == "auto"
    assert args.detector_device == "cpu"
    assert args.recognizer_device == "cuda"


def test_realign_layout_gt_cli_requires_only_cached_layout_root() -> None:
    args = build_parser().parse_args(
        [
            "realign-layout-gt",
            "--layout-gt",
            "work/layout_gt/t5",
        ]
    )

    assert args.layout_gt == Path("work/layout_gt/t5")
    assert args.max_alignment_boxes == 12


def test_clean_layout_gt_cli_uses_conservative_profile_by_default() -> None:
    args = build_parser().parse_args(
        [
            "clean-layout-gt",
            "--layout-gt",
            "work/layout_gt/t5",
            "--output",
            "work/layout_gt/t5_clean",
        ]
    )

    assert args.layout_gt == Path("work/layout_gt/t5")
    assert args.output == Path("work/layout_gt/t5_clean")
    assert args.profile == "conservative"


def test_requested_layout_training_cli_can_derive_data_and_gt_from_layout_manifest() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--stage",
            "layout",
            "--model",
            "layoutlmv3",
            "--layout-training-mode",
            "linear_probe",
            "--layout-gt",
            "work/layout_gt/t5",
            "--split-manifest",
            "GT/splits/split_v1.json",
            "--output",
            "models/finetuned/layoutlmv3_linear_probe",
        ]
    )
    assert args.data is None
    assert args.gt is None
    assert args.layout_gt == Path("work/layout_gt/t5")


def test_requested_direct_end_to_end_cli_accepts_trained_layout_checkpoint() -> None:
    args = build_parser().parse_args(
        [
            "experiment",
            "--pipeline",
            "paddleocr",
            "vietocr",
            "layoutlmv3",
            "--layout-checkpoint",
            "models/finetuned/layoutlmv3_full_finetune/best",
            "--detector-device",
            "cpu",
            "--recognizer-device",
            "cuda",
            "--layout-device",
            "cuda",
            "--data",
            "data/t5",
            "--gt",
            "GT",
            "--split",
            "test",
            "--output",
            "outputs/experiments/t5_full_finetune",
        ]
    )
    assert args.layout_checkpoint == Path("models/finetuned/layoutlmv3_full_finetune/best")
    assert args.split == "test"
    assert args.detector_device == "cpu"
    assert args.recognizer_device == "cuda"
    assert args.layout_device == "cuda"


def test_final_gt_values_are_loaded_only_after_all_test_inference(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from invoice_ocr.experiments import evaluate_pipeline as module

    data = tmp_path / "data" / "t5"
    gt = tmp_path / "GT"
    canonical = InvoiceBatch(
        invoice_count=1,
        invoices=[InvoiceDocument(page_number=1)],
    ).model_dump(mode="json")
    for index in range(20):
        source = data / "synthetic-folder" / f"invoice-{index:02d}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(index, index, index)).save(source)
        _write_json(
            gt / "final" / "t5" / "synthetic-folder" / f"invoice-{index:02d}.json",
            canonical,
        )
    split = gt / "splits" / "split_v1.json"
    locked = create_locked_split(data, gt, split, seed=42)
    assert len(locked.test_document_ids) == 2
    events: list[str] = []
    adapter_devices: dict[str, str] = {}
    oriented_pages: list[int] = []

    class FakeDetector:
        def __init__(self, _model_root: Path, device: str, _checkpoint: Path | None) -> None:
            adapter_devices["detector"] = device

        def prepare(self) -> None:
            pass

        def detect(self, _page: Any) -> list[Any]:
            return []

    class FakeRecognizer:
        def __init__(self, _model_root: Path, device: str, _checkpoint: Path | None) -> None:
            adapter_devices["recognizer"] = device

        def prepare(self) -> None:
            pass

        def recognize(self, _page: Any, _regions: list[Any]) -> list[Any]:
            return []

    class FakeLayout:
        provides_invoice_labels = True

        def __init__(self, _model_root: Path, device: str, _checkpoint: Path | None) -> None:
            adapter_devices["layout"] = device

        def prepare(self) -> None:
            pass

        def extract(self, _page: Any, _regions: list[Any]) -> tuple[list[Any], list[Any]]:
            events.append("inference")
            return [], []

    monkeypatch.setitem(module.DETECTORS, "paddleocr", FakeDetector)
    monkeypatch.setitem(module.RECOGNIZERS, "vietocr", FakeRecognizer)
    monkeypatch.setitem(module.LAYOUT_ADAPTERS, "layoutlmv3", FakeLayout)
    monkeypatch.setattr(module, "resolve_device", lambda requested: requested)

    def orient_page(
        page: Any,
        detector: FakeDetector,
        _recognizer: FakeRecognizer,
    ) -> tuple[Any, list[Any]]:
        oriented_pages.append(page.page_index)
        return page, detector.detect(page)

    monkeypatch.setattr(module, "auto_orient_page", orient_page)
    original_load = module._load_object

    def tracked_load(path: Path) -> dict[str, Any]:
        if "final" in path.parts and path.suffix == ".json":
            assert events.count("inference") == len(locked.test_document_ids)
            events.append("gt")
        return original_load(path)

    monkeypatch.setattr(module, "_load_object", tracked_load)
    output = tmp_path / "outputs" / "end-to-end"
    evaluate_pipeline(
        PipelineEvaluationRequest(
            pipeline=PipelineSelection("paddleocr", "vietocr", "layoutlmv3"),
            checkpoints={"detector": None, "recognizer": None, "layout": None},
            run_kind="finetuned",
            data_root=data,
            gt_root=gt,
            split_manifest=split,
            output_dir=output,
            work_root=tmp_path / "work",
            device="cpu",
            detector_device="cpu",
            recognizer_device="cuda",
            layout_device="cuda",
        )
    )
    assert adapter_devices == {
        "detector": "cpu",
        "recognizer": "cuda",
        "layout": "cuda",
    }
    assert oriented_pages == [0, 0]
    assert events[: len(locked.test_document_ids)] == ["inference", "inference"]
    assert events[len(locked.test_document_ids) :] == ["gt", "gt"]


def test_locked_layout_training_reads_pseudo_annotations_and_excludes_test(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data" / "t5"
    gt = tmp_path / "GT"
    canonical = {
        "document_type": "VAT_INVOICE_BATCH",
        "invoice_count": 0,
        "invoices": [],
    }
    for index in range(6):
        source = data / "synthetic-folder" / f"invoice-{index}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(index, index, index)).save(source)
        _write_json(
            gt / "final" / "t5" / "synthetic-folder" / f"invoice-{index}.json",
            canonical,
        )
    split_path = gt / "splits" / "t5_split.json"
    split = create_locked_split(data, gt, split_path, seed=42)
    layout_gt = tmp_path / "work" / "layout_gt" / "t5"
    for document in discover_documents(data):
        _write_json(
            layout_gt / "layout" / f"{document.document_id}.json",
            {
                "document_id": document.document_id,
                "pages": [
                    {
                        "page_index": 0,
                        "image_path": "images/synthetic.png",
                        "tokens": ["Synthetic"],
                        "boxes": [[0, 0, 100, 100]],
                        "labels": ["O"],
                    }
                ],
            },
        )

    class FakeBackend:
        def __init__(self) -> None:
            self.train_paths: list[Path] = []
            self.validation_paths: list[Path] = []

        def train(
            self,
            request: LockedTrainingRequest,
            train_annotations: list[Path],
            validation_annotations: list[Path],
            _device: str,
        ) -> TrainingBackendResult:
            self.train_paths = train_annotations
            self.validation_paths = validation_annotations
            checkpoint = request.output_dir / "backend" / "checkpoint-1"
            _write_json(checkpoint / "synthetic.json", {"synthetic": True})
            return TrainingBackendResult(
                candidates=[
                    CheckpointCandidate(
                        path=checkpoint,
                        epoch=1,
                        metric_value=0.5,
                        evaluated_split="validation",
                    )
                ],
                epochs_completed=1.0,
                time_per_epoch_seconds=0.1,
                early_stopping_epoch=None,
                log_history=[],
                training_samples=len(train_annotations),
                validation_samples=len(validation_annotations),
            )

    backend = FakeBackend()
    output = tmp_path / "models" / "linear"
    train_model_locked(
        LockedTrainingRequest(
            stage="layout",
            model="layoutlmv3",
            checkpoint_source="pretrained",
            data_root=data,
            gt_root=gt,
            layout_gt_root=layout_gt,
            split_manifest=split_path,
            output_dir=output,
            device="cpu",
            layout_training_mode="linear_probe",
        ),
        backend=backend,
    )
    assert {path.stem for path in backend.train_paths} == set(split.train_document_ids)
    assert {path.stem for path in backend.validation_paths} == set(split.validation_document_ids)
    assert not set(split.test_document_ids) & {
        path.stem for path in backend.train_paths + backend.validation_paths
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["test_set_used_for_training"] is False

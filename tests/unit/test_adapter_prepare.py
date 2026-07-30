from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from invoice_ocr.adapters.detectors.paddleocr import PaddleOCRDetector
from invoice_ocr.adapters.recognizers.paddleocr import PaddleOCRRecognizer
from invoice_ocr.exceptions import CheckpointUnavailableError


def test_paddle_adapters_prepare_and_cache_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=factory),
    )
    detector_checkpoint = tmp_path / "paddleocr" / "detector"
    recognizer_checkpoint = tmp_path / "paddleocr" / "recognizer"
    for checkpoint in (detector_checkpoint, recognizer_checkpoint):
        checkpoint.mkdir(parents=True)
        (checkpoint / "inference.pdmodel").write_bytes(b"synthetic model")
        (checkpoint / "inference.pdiparams").write_bytes(b"synthetic parameters")
    detector = PaddleOCRDetector(tmp_path, device="cpu")
    recognizer = PaddleOCRRecognizer(tmp_path, device="cpu")
    detector.prepare()
    detector.prepare()
    recognizer.prepare()
    recognizer.prepare()
    assert len(calls) == 2
    assert "det" not in calls[0]
    assert calls[0]["det_model_dir"] == str(detector_checkpoint)
    assert calls[1]["det"] is False
    assert calls[1]["rec_model_dir"] == str(recognizer_checkpoint)


def test_paddle_adapter_refuses_package_auto_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=factory),
    )
    detector = PaddleOCRDetector(tmp_path, device="cpu")

    with pytest.raises(CheckpointUnavailableError, match="checkpoint is incomplete"):
        detector.prepare()

    assert calls == []

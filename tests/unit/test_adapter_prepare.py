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
    recognizer_calls: list[dict[str, object]] = []
    detector_calls: list[SimpleNamespace] = []

    def recognizer_factory(**kwargs: object) -> object:
        recognizer_calls.append(kwargs)
        return object()

    def detector_factory(params: SimpleNamespace) -> object:
        detector_calls.append(params)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=recognizer_factory),
    )
    monkeypatch.setitem(
        sys.modules,
        "paddleocr.paddleocr",
        SimpleNamespace(parse_args=lambda **_kwargs: SimpleNamespace()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.infer.predict_det",
        SimpleNamespace(TextDetector=detector_factory),
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
    assert len(detector_calls) == 1
    assert detector_calls[0].det is True
    assert detector_calls[0].rec is False
    assert detector_calls[0].det_model_dir == str(detector_checkpoint)
    assert detector_calls[0].ir_optim is False
    assert detector_calls[0].use_tensorrt is False
    assert detector_calls[0].enable_mkldnn is False
    assert recognizer_calls == [
        {
            "det": False,
            "use_angle_cls": False,
            "use_gpu": False,
            "show_log": False,
            "lang": "vi",
            "rec_model_dir": str(recognizer_checkpoint),
        }
    ]


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

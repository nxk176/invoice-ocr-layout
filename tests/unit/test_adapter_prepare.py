from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from invoice_ocr.adapters.detectors.paddleocr import PaddleOCRDetector
from invoice_ocr.adapters.recognizers.paddleocr import PaddleOCRRecognizer


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
    detector = PaddleOCRDetector(tmp_path, device="cpu")
    recognizer = PaddleOCRRecognizer(tmp_path, device="cpu")
    detector.prepare()
    detector.prepare()
    recognizer.prepare()
    recognizer.prepare()
    assert len(calls) == 2
    assert "det" not in calls[0]
    assert calls[1]["det"] is False

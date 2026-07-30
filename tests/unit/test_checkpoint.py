from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from invoice_ocr.adapters.recognizers.vietocr import VietOCRRecognizer
from invoice_ocr.exceptions import CheckpointUnavailableError


def test_missing_checkpoint_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = ModuleType("vietocr")
    tool = ModuleType("vietocr.tool")
    config = ModuleType("vietocr.tool.config")
    predictor = ModuleType("vietocr.tool.predictor")

    class FakeCfg:
        @staticmethod
        def load_config_from_name(name: str) -> dict[str, object]:
            return {"cnn": {}}

    class FakePredictor:
        pass

    config.Cfg = FakeCfg  # type: ignore[attr-defined]
    predictor.Predictor = FakePredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vietocr", package)
    monkeypatch.setitem(sys.modules, "vietocr.tool", tool)
    monkeypatch.setitem(sys.modules, "vietocr.tool.config", config)
    monkeypatch.setitem(sys.modules, "vietocr.tool.predictor", predictor)

    adapter = VietOCRRecognizer(tmp_path)
    with pytest.raises(CheckpointUnavailableError, match="checkpoint not found"):
        adapter._create_predictor()

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from invoice_ocr.adapters.detectors.dbnet import DBNetDetector
from invoice_ocr.exceptions import DependencyUnavailableError


def test_missing_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    adapter = DBNetDetector(tmp_path)
    with pytest.raises(DependencyUnavailableError, match="official MhLiao/DB checkout"):
        adapter._validate_runtime()

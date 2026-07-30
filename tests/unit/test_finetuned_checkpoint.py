from pathlib import Path

import pytest

from invoice_ocr.adapters.layout.layoutlmv3_complete import LayoutLMv3Adapter
from invoice_ocr.exceptions import CheckpointUnavailableError


def test_layoutlmv3_finds_framework_default_training_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "outputs" / "training" / "layout-layoutlmv3" / "invoice-best"
    checkpoint.mkdir(parents=True)
    adapter = LayoutLMv3Adapter(tmp_path / "models")
    assert adapter.resolve_checkpoint() == checkpoint


def test_layoutlmv3_missing_finetuned_checkpoint_is_explicit(tmp_path: Path) -> None:
    adapter = LayoutLMv3Adapter(tmp_path / "models")
    with pytest.raises(CheckpointUnavailableError, match="base checkpoint does not know"):
        adapter.resolve_checkpoint()

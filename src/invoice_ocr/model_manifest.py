"""Read exact upstream revisions for persisted stage provenance."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

MANIFEST_BY_ADAPTER = {
    ("detector", "paddleocr"): "paddleocr-detector",
    ("detector", "dbnet"): "dbnet",
    ("detector", "dbnetpp"): "dbnetpp",
    ("recognizer", "paddleocr"): "paddleocr-recognizer",
    ("recognizer", "vietocr"): "vietocr",
    ("layout", "layoutlmv3"): "layoutlmv3-base",
    ("layout", "vi_layoutxlm"): "vi-layoutxlm-base",
}


@lru_cache
def load_adapter_manifest(stage: str, adapter_name: str) -> dict[str, Any]:
    manifest_name = MANIFEST_BY_ADAPTER[(stage, adapter_name)]
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "configs" / "models" / f"{manifest_name}.yaml"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"invalid model manifest: {path}")
    return loaded


def adapter_revision(stage: str, adapter_name: str) -> str:
    return str(load_adapter_manifest(stage, adapter_name)["revision"])


"""Locked pretrained-vs-fine-tuned experiment protocol."""

from invoice_ocr.experiments.contracts import LockedSplitManifest
from invoice_ocr.experiments.split import load_locked_split

__all__ = ["LockedSplitManifest", "load_locked_split"]

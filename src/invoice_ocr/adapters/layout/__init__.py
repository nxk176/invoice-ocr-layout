"""Layout/KIE adapters."""

from invoice_ocr.adapters.layout.base import LayoutAdapter
from invoice_ocr.adapters.layout.layoutlmv3_complete import LayoutLMv3Adapter
from invoice_ocr.adapters.layout.vi_layoutxlm import VILayoutXLMAdapter

LAYOUT_ADAPTERS: dict[str, type[LayoutAdapter]] = {
    "layoutlmv3": LayoutLMv3Adapter,
    "vi_layoutxlm": VILayoutXLMAdapter,
}

__all__ = ["LAYOUT_ADAPTERS", "LayoutAdapter"]

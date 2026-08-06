"""Layout/KIE adapters."""

from invoice_ocr.adapters.layout.base import LayoutAdapter
from invoice_ocr.adapters.layout.layoutlmv3_complete import LayoutLMv3Adapter
from invoice_ocr.adapters.layout.layoutlmv3_pretrained import LayoutLMv3PretrainedAdapter
from invoice_ocr.adapters.layout.vi_layoutxlm import VILayoutXLMAdapter

LAYOUT_ADAPTERS: dict[str, type[LayoutAdapter]] = {
    "layoutlmv3": LayoutLMv3Adapter,
    "layoutlmv3_pretrained": LayoutLMv3PretrainedAdapter,
    "vi_layoutxlm": VILayoutXLMAdapter,
}

__all__ = ["LAYOUT_ADAPTERS", "LayoutAdapter"]

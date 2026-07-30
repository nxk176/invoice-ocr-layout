"""Text recognizer adapters."""

from invoice_ocr.adapters.recognizers.base import RecognizerAdapter
from invoice_ocr.adapters.recognizers.paddleocr import PaddleOCRRecognizer
from invoice_ocr.adapters.recognizers.vietocr import VietOCRRecognizer

RECOGNIZERS: dict[str, type[RecognizerAdapter]] = {
    "paddleocr": PaddleOCRRecognizer,
    "vietocr": VietOCRRecognizer,
}

__all__ = ["RECOGNIZERS", "RecognizerAdapter"]

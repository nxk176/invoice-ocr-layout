"""Text detector adapters."""

from invoice_ocr.adapters.detectors.base import DetectorAdapter
from invoice_ocr.adapters.detectors.dbnet import DBNetDetector
from invoice_ocr.adapters.detectors.dbnetpp import DBNetPPDetector
from invoice_ocr.adapters.detectors.paddleocr import PaddleOCRDetector

DETECTORS: dict[str, type[DetectorAdapter]] = {
    "paddleocr": PaddleOCRDetector,
    "dbnet": DBNetDetector,
    "dbnetpp": DBNetPPDetector,
}

__all__ = ["DETECTORS", "DetectorAdapter"]

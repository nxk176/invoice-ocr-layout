"""Pseudo LayoutLM annotations derived from pretrained OCR and canonical final GT."""

from invoice_ocr.layout_gt.alignment import (
    AlignmentResult,
    GroundTruthField,
    OCRRegion,
    align_ground_truth_fields,
    flatten_canonical_ground_truth,
    normalize_layout_bbox,
)
from invoice_ocr.layout_gt.index import (
    FinalGroundTruthIndex,
    FinalGroundTruthTarget,
    build_final_ground_truth_index,
    load_final_ground_truth_index,
)

__all__ = [
    "AlignmentResult",
    "FinalGroundTruthIndex",
    "FinalGroundTruthTarget",
    "GroundTruthField",
    "OCRRegion",
    "align_ground_truth_fields",
    "build_final_ground_truth_index",
    "flatten_canonical_ground_truth",
    "load_final_ground_truth_index",
    "normalize_layout_bbox",
]

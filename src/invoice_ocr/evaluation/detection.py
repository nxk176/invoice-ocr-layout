"""Text detection metrics using axis-aligned annotation boxes."""

from __future__ import annotations

from dataclasses import dataclass

from invoice_ocr.contracts import BoundingBox


def box_iou(left: BoundingBox, right: BoundingBox) -> float:
    x_min = max(left.x_min, right.x_min)
    y_min = max(left.y_min, right.y_min)
    x_max = min(left.x_max, right.x_max)
    y_max = min(left.y_max, right.y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    left_area = (left.x_max - left.x_min) * (left.y_max - left.y_min)
    right_area = (right.x_max - right.x_min) * (right.y_max - right.y_min)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class DetectionMetrics:
    mean_iou: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def detection_metrics(
    predicted: list[BoundingBox],
    expected: list[BoundingBox],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be in 0..1")
    pairs = sorted(
        (
            (box_iou(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predicted)
            for target_index, target in enumerate(expected)
        ),
        reverse=True,
    )
    matched_predicted: set[int] = set()
    matched_expected: set[int] = set()
    matched_ious: list[float] = []
    for iou, prediction_index, target_index in pairs:
        if iou < iou_threshold:
            break
        if prediction_index in matched_predicted or target_index in matched_expected:
            continue
        matched_predicted.add(prediction_index)
        matched_expected.add(target_index)
        matched_ious.append(iou)
    true_positives = len(matched_ious)
    false_positives = len(predicted) - true_positives
    false_negatives = len(expected) - true_positives
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionMetrics(
        mean_iou=sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )

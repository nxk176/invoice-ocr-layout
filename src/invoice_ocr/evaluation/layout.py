"""Entity and relation set metrics for KIE evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T", bound=tuple[object, ...])


@dataclass(frozen=True)
class SetMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def set_metrics(predicted: set[T], expected: set[T]) -> SetMetrics:
    true_positives = len(predicted & expected)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SetMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )

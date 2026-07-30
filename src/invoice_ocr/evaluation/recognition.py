"""OCR transcription metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass

NUMERIC_PATTERN = re.compile(r"^[\d.,+\-/%\s]+$")


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class RecognitionMetrics:
    cer: float
    wer: float
    exact_match: float
    numeric_field_accuracy: float | None


def recognition_metrics(predicted: list[str], expected: list[str]) -> RecognitionMetrics:
    if len(predicted) != len(expected):
        raise ValueError("prediction and ground-truth transcription counts differ")
    expected_chars = sum(len(value) for value in expected)
    expected_words = sum(len(value.split()) for value in expected)
    char_errors = sum(
        edit_distance(list(p), list(e)) for p, e in zip(predicted, expected, strict=False)
    )
    word_errors = sum(
        edit_distance(p.split(), e.split()) for p, e in zip(predicted, expected, strict=False)
    )
    exact = sum(p == e for p, e in zip(predicted, expected, strict=False))
    numeric_pairs = [
        (p, e)
        for p, e in zip(predicted, expected, strict=False)
        if NUMERIC_PATTERN.fullmatch(e.strip())
    ]
    return RecognitionMetrics(
        cer=char_errors / expected_chars if expected_chars else 0.0,
        wer=word_errors / expected_words if expected_words else 0.0,
        exact_match=exact / len(expected) if expected else 0.0,
        numeric_field_accuracy=(
            sum(p == e for p, e in numeric_pairs) / len(numeric_pairs) if numeric_pairs else None
        ),
    )

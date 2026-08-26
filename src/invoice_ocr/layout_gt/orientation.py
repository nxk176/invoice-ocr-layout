"""Pretrained-OCR page orientation for pseudo-layout annotation."""

from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean
from typing import Literal

from PIL import Image

from invoice_ocr.adapters.detectors.base import DetectorAdapter
from invoice_ocr.adapters.recognizers.base import RecognizerAdapter
from invoice_ocr.contracts import DetectionRegion, DocumentPage, OrientationMetadata

LOGGER = logging.getLogger("invoice_ocr")
_MIN_ORIENTATION_REGIONS = 8
_VERTICAL_TRIGGER_FRACTION = 0.7
_ORIENTATION_SAMPLE_SIZE = 16


def _region_dimensions(region: DetectionRegion) -> tuple[float, float]:
    return (
        region.bbox.x_max - region.bbox.x_min,
        region.bbox.y_max - region.bbox.y_min,
    )


def vertical_region_fraction(regions: list[DetectionRegion]) -> float:
    """Return the fraction of boxes whose long axis is clearly vertical."""
    if not regions:
        return 0.0
    vertical = sum(height > width * 1.25 for width, height in map(_region_dimensions, regions))
    return vertical / len(regions)


def _recognition_orientation_score(
    page: DocumentPage,
    detections: list[DetectionRegion],
    recognizer: RecognizerAdapter,
) -> float:
    if not detections:
        return 0.0
    sample = sorted(
        detections,
        key=lambda region: (
            -max(_region_dimensions(region)),
            region.bbox.y_min,
            region.bbox.x_min,
        ),
    )[:_ORIENTATION_SAMPLE_SIZE]
    recognized = recognizer.recognize(page, sample)
    if not recognized:
        return 0.0
    scores: list[float] = []
    for region in recognized:
        character_count = sum(character.isalnum() for character in region.text)
        length_quality = min(character_count / 8, 1.0)
        scores.append(region.confidence * (0.2 + 0.8 * length_quality))
    horizontal_fraction = 1.0 - vertical_region_fraction(detections)
    return mean(scores) + horizontal_fraction * 0.05


def _rotate_page_candidate(
    page: DocumentPage,
    rotation: Literal[90, 270],
) -> DocumentPage:
    transpose = {
        90: Image.Transpose.ROTATE_90,
        270: Image.Transpose.ROTATE_270,
    }[rotation]
    source_path = Path(page.image_path)
    target_path = source_path.with_name(
        f"{source_path.stem}.orientation-{rotation}{source_path.suffix}"
    )
    with Image.open(source_path) as source:
        rotated = source.transpose(transpose)
        rotated.save(target_path)
        width, height = rotated.size
    return page.model_copy(
        update={
            "image_path": str(target_path),
            "width": width,
            "height": height,
            "orientation": OrientationMetadata(
                rotation_degrees=rotation,
                method="pretrained_ocr_orientation_candidate",
            ),
        }
    )


def _remove_candidate_images(candidates: list[DocumentPage]) -> None:
    for candidate in candidates:
        path = Path(candidate.image_path)
        if path.is_file():
            path.unlink()


def auto_orient_page(
    page: DocumentPage,
    detector: DetectorAdapter,
    recognizer: RecognizerAdapter,
) -> tuple[DocumentPage, list[DetectionRegion]]:
    """Rotate a predominantly vertical OCR page without consulting field GT."""
    initial_detections = detector.detect(page)
    vertical_fraction = vertical_region_fraction(initial_detections)
    if (
        len(initial_detections) < _MIN_ORIENTATION_REGIONS
        or vertical_fraction < _VERTICAL_TRIGGER_FRACTION
    ):
        return page, initial_detections

    candidates: list[DocumentPage] = []
    try:
        scored: list[tuple[float, int, DocumentPage, list[DetectionRegion]]] = []
        for rotation in (90, 270):
            candidate = _rotate_page_candidate(page, rotation)
            candidates.append(candidate)
            detections = detector.detect(candidate)
            score = _recognition_orientation_score(candidate, detections, recognizer)
            scored.append((score, -rotation, candidate, detections))
        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        best_score, _, selected_page, selected_detections = scored[0]
        second_score = scored[1][0]
        confidence = min(1.0, max(0.0, 0.5 + best_score - second_score))
        selected_path = Path(selected_page.image_path)
        original_path = Path(page.image_path)
        unselected = [
            candidate
            for candidate in candidates
            if candidate.image_path != selected_page.image_path
        ]
        _remove_candidate_images(unselected)
        selected_path.replace(original_path)
        selected_page = selected_page.model_copy(
            update={
                "image_path": str(original_path),
                "orientation": OrientationMetadata(
                    rotation_degrees=selected_page.orientation.rotation_degrees,
                    confidence=confidence,
                    method="pretrained_ocr_orientation",
                ),
            }
        )
        LOGGER.info(
            "Auto-oriented %s page %d by %d degrees: "
            "vertical_fraction=%.3f score=%.3f alternate_score=%.3f",
            page.document_id,
            page.page_index + 1,
            selected_page.orientation.rotation_degrees,
            vertical_fraction,
            best_score,
            second_score,
        )
        return selected_page, selected_detections
    except Exception:
        _remove_candidate_images(candidates)
        raise

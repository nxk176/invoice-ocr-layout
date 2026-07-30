"""Conservative deskew estimation and image rotation."""

from __future__ import annotations

from pathlib import Path

from invoice_ocr.exceptions import DependencyUnavailableError


def deskew_image(source: Path, destination: Path, max_abs_angle: float = 10.0) -> float:
    """Estimate text-line skew with OpenCV and save a corrected image."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise DependencyUnavailableError(
            "Deskew preprocessing requires numpy and opencv-python-headless."
        ) from exc
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"cannot read image for deskew: {source}")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coordinates = np.column_stack(np.where(grayscale < 250))
    angle = 0.0 if len(coordinates) < 10 else float(cv2.minAreaRect(coordinates)[-1])
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > max_abs_angle:
        angle = 0.0
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    corrected = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), corrected):
        raise OSError(f"failed to write deskewed image: {destination}")
    return angle


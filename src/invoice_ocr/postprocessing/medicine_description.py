"""Conservative extraction from a printed medicine description."""

from __future__ import annotations

import re

STRENGTH_PATTERN = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|µg|ml|%|IU|UI)(?:\s*/\s*\d+\s*ml)?)",
    re.IGNORECASE,
)
COUNTRY_PATTERN = re.compile(
    r"(?:NSX|nước\s+sản\s+xuất|xuất\s+xứ)\s*:\s*([^;,|]+)",
    re.IGNORECASE,
)
MANUFACTURER_PATTERN = re.compile(
    r"(?:nhà\s+sản\s+xuất|manufacturer)\s*:\s*([^;,|]+)",
    re.IGNORECASE,
)


def parse_medicine_description(raw_description: str | None) -> dict[str, str | None]:
    """Return only fields explicitly supported by labels or strong printed patterns."""
    result: dict[str, str | None] = {
        "medicine_name": None,
        "strength": None,
        "manufacturer": None,
        "country_of_manufacture": None,
    }
    if raw_description is None or not raw_description.strip():
        return result
    text = raw_description.strip()
    strength = STRENGTH_PATTERN.search(text)
    country = COUNTRY_PATTERN.search(text)
    manufacturer = MANUFACTURER_PATTERN.search(text)
    result["strength"] = strength.group(1).strip() if strength else None
    result["country_of_manufacture"] = country.group(1).strip() if country else None
    result["manufacturer"] = manufacturer.group(1).strip() if manufacturer else None
    label_starts = [match.start() for match in (country, manufacturer) if match is not None]
    labeled_tail = min(label_starts or [len(text)])
    name_candidate = text[:labeled_tail]
    if strength:
        name_candidate = name_candidate[: strength.start()]
    name_candidate = name_candidate.strip(" ,;-")
    result["medicine_name"] = name_candidate or None
    return result

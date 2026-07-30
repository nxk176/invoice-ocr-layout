"""Conservative date normalization."""

from __future__ import annotations

import re
from datetime import date

DATE_PATTERN = re.compile(r"(?<!\d)(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})(?!\d)")
VIETNAMESE_DATE_PATTERN = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE,
)


def normalize_date(value: str | None) -> str | None:
    """Normalize an unambiguous printed date to YYYY-MM-DD; otherwise return null."""
    if value is None:
        return None
    text = value.strip()
    vietnamese = VIETNAMESE_DATE_PATTERN.search(text)
    if vietnamese:
        day, month, year = map(int, vietnamese.groups())
    else:
        match = DATE_PATTERN.search(text)
        if not match:
            return None
        first, second, third = match.groups()
        if len(first) == 4:
            year, month, day = int(first), int(second), int(third)
        elif len(third) == 4:
            day, month, year = int(first), int(second), int(third)
        else:
            return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None

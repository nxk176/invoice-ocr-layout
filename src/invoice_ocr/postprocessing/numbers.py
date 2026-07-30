"""Vietnamese/English printed number parsing without arithmetic correction."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

NUMBER_CHARS = re.compile(r"[^0-9,.\-+]")


def parse_vietnamese_number(value: str | int | float | Decimal | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    text = NUMBER_CHARS.sub("", value.strip())
    if not text or text in {"-", "+", ".", ","}:
        return None
    sign = ""
    if text[0] in "+-":
        sign, text = text[0], text[1:]
    if "+" in text or "-" in text:
        return None
    last_dot = text.rfind(".")
    last_comma = text.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        decimal_separator = "." if last_dot > last_comma else ","
        thousands_separator = "," if decimal_separator == "." else "."
        normalized = text.replace(thousands_separator, "").replace(decimal_separator, ".")
    elif "." in text or "," in text:
        separator = "." if "." in text else ","
        groups = text.split(separator)
        if len(groups) > 2:
            if any(len(group) != 3 for group in groups[1:]):
                return None
            normalized = "".join(groups)
        elif len(groups[1]) == 3 and len(groups[0]) >= 1:
            normalized = "".join(groups)
        else:
            normalized = ".".join(groups)
    else:
        normalized = text
    try:
        return Decimal(f"{sign}{normalized}")
    except InvalidOperation:
        return None

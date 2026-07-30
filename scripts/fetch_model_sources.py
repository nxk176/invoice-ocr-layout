"""Fetch required official source repositories at exact audited revisions."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from invoice_ocr.model_sources import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

"""Download declared public base checkpoints without downloading invoice weights."""

from __future__ import annotations

import sys

from model_downloader import run

if __name__ == "__main__":
    sys.exit(run())

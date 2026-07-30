from __future__ import annotations

import subprocess
from pathlib import Path


def test_git_tracks_no_sensitive_binary_or_large_fixture() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    tracked = [Path(value.decode()) for value in completed.stdout.split(b"\0") if value]
    forbidden = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".pth",
        ".pt",
        ".bin",
        ".onnx",
        ".pdparams",
        ".safetensors",
    }
    violations = [path for path in tracked if path.suffix.casefold() in forbidden]
    large = [path for path in tracked if path.is_file() and path.stat().st_size > 10 * 1024 * 1024]
    assert violations == []
    assert large == []

from __future__ import annotations

import subprocess
from pathlib import Path

from invoice_ocr.model_sources import (
    SourceSpec,
    fetch_source_checkout,
    load_source_specs,
    verify_source_checkout,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _create_upstream(tmp_path: Path) -> tuple[Path, str, str]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", str(upstream)], check=True, capture_output=True, text=True)
    _git(upstream, "config", "user.email", "synthetic@example.invalid")
    _git(upstream, "config", "user.name", "Synthetic Test")
    (upstream / "model.py").write_text("VERSION = 1\n", encoding="utf-8")
    _git(upstream, "add", "model.py")
    _git(upstream, "commit", "-m", "first")
    first = _git(upstream, "rev-parse", "HEAD")
    (upstream / "model.py").write_text("VERSION = 2\n", encoding="utf-8")
    _git(upstream, "add", "model.py")
    _git(upstream, "commit", "-m", "second")
    second = _git(upstream, "rev-parse", "HEAD")
    return upstream, first, second


def _spec(upstream: Path, revision: str) -> SourceSpec:
    return SourceSpec(
        name="synthetic",
        backends=("synthetic",),
        official_repository=str(upstream.resolve()),
        revision=revision,
        local_path=Path("Synthetic"),
        required=True,
        required_for=("test",),
        reason="synthetic test checkout",
        license="test-only",
    )


def test_source_revision_config_parses_exact_commits() -> None:
    specs = load_source_specs(PROJECT_ROOT / "configs" / "models" / "source_revisions.yaml")

    assert set(specs) == {"paddleocr", "dbnet", "vietocr"}
    assert all(len(spec.revision) == 40 for spec in specs.values())
    assert all(
        "main" not in spec.revision and "master" not in spec.revision for spec in specs.values()
    )
    assert specs["vietocr"].required is False


def test_source_fetch_dry_run_does_not_create_checkout(tmp_path: Path) -> None:
    upstream, revision, _ = _create_upstream(tmp_path)
    external = tmp_path / "external"

    status = fetch_source_checkout(_spec(upstream, revision), external, dry_run=True)

    assert status.status == "WOULD_CLONE"
    assert not (external / "Synthetic").exists()


def test_source_checkout_fetch_and_verification(tmp_path: Path) -> None:
    upstream, revision, _ = _create_upstream(tmp_path)
    external = tmp_path / "external"
    spec = _spec(upstream, revision)

    fetched = fetch_source_checkout(spec, external)
    verified = verify_source_checkout(spec, external)

    assert fetched.ready
    assert verified.status == "READY"
    assert verified.actual_commit == revision
    assert verified.actual_remote is not None


def test_dirty_checkout_is_not_overwritten(tmp_path: Path) -> None:
    upstream, revision, _ = _create_upstream(tmp_path)
    external = tmp_path / "external"
    spec = _spec(upstream, revision)
    assert fetch_source_checkout(spec, external).ready
    checkout = external / "Synthetic"
    marker = checkout / "local-change.txt"
    marker.write_text("preserve me\n", encoding="utf-8")
    head_before = _git(checkout, "rev-parse", "HEAD")

    status = fetch_source_checkout(spec, external)

    assert status.status == "DIRTY"
    assert "refusing to overwrite" in status.reason
    assert marker.read_text(encoding="utf-8") == "preserve me\n"
    assert _git(checkout, "rev-parse", "HEAD") == head_before


def test_wrong_source_commit_is_reported_without_checkout_change(tmp_path: Path) -> None:
    upstream, first, second = _create_upstream(tmp_path)
    external = tmp_path / "external"
    spec = _spec(upstream, first)
    assert fetch_source_checkout(spec, external).ready
    checkout = external / "Synthetic"
    _git(checkout, "checkout", "--detach", second)

    status = fetch_source_checkout(spec, external)

    assert status.status == "WRONG_COMMIT"
    assert status.actual_commit == second
    assert _git(checkout, "rev-parse", "HEAD") == second

"""Safe revision-pinned external source checkout and verification."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models" / "source_revisions.yaml"
DEFAULT_EXTERNAL_ROOT = PROJECT_ROOT / "external"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    backends: tuple[str, ...]
    official_repository: str
    revision: str
    local_path: Path
    required: bool
    required_for: tuple[str, ...]
    reason: str
    license: str
    tag: str | None = None


@dataclass(frozen=True)
class SourceStatus:
    name: str
    path: Path
    required: bool
    found: bool
    ready: bool
    expected_commit: str
    actual_commit: str | None
    expected_remote: str
    actual_remote: str | None
    dirty: bool | None
    status: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.name,
            "path": str(self.path),
            "required": self.required,
            "found": self.found,
            "ready": self.ready,
            "expected_commit": self.expected_commit,
            "actual_commit": self.actual_commit,
            "expected_remote": self.expected_remote,
            "actual_remote": self.actual_remote,
            "dirty": self.dirty,
            "status": self.status,
            "reason": self.reason,
        }


def _as_string_list(value: Any, field: str, source: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"source '{source}' field '{field}' must be a string list")
    return tuple(value)


def load_source_specs(path: Path = DEFAULT_CONFIG) -> dict[str, SourceSpec]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("sources"), dict):
        raise ValueError(f"source revision config must contain a sources mapping: {path}")
    specs: dict[str, SourceSpec] = {}
    for name, raw in loaded["sources"].items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ValueError(f"invalid source entry in {path}: {name!r}")
        revision = str(raw.get("revision", ""))
        if not COMMIT_PATTERN.fullmatch(revision):
            raise ValueError(f"source '{name}' revision must be an exact lowercase Git commit")
        local_path = Path(str(raw.get("local_path", "")))
        if local_path.is_absolute() or ".." in local_path.parts or not local_path.parts:
            raise ValueError(f"source '{name}' local_path must stay below external/")
        repository = str(raw.get("official_repository", ""))
        if not repository:
            raise ValueError(f"source '{name}' official_repository is required")
        specs[name] = SourceSpec(
            name=name,
            backends=_as_string_list(raw.get("backends"), "backends", name),
            official_repository=repository,
            revision=revision,
            local_path=local_path,
            required=bool(raw.get("required")),
            required_for=_as_string_list(raw.get("required_for"), "required_for", name),
            reason=str(raw.get("reason", "")),
            license=str(raw.get("license", "")),
            tag=str(raw["tag"]) if raw.get("tag") is not None else None,
        )
    return specs


def _git(path: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _normalize_remote(value: str) -> str:
    normalized = value.strip().replace("\\", "/").removesuffix("/").removesuffix(".git")
    if "://" not in normalized and not normalized.startswith("git@"):
        return Path(normalized).resolve().as_posix().casefold()
    return normalized.casefold()


def verify_source_checkout(spec: SourceSpec, external_root: Path) -> SourceStatus:
    target = external_root / spec.local_path
    if not spec.required and not target.exists():
        return SourceStatus(
            spec.name,
            target,
            False,
            False,
            True,
            spec.revision,
            None,
            spec.official_repository,
            None,
            None,
            "NOT_REQUIRED",
            spec.reason,
        )
    if not target.exists():
        return SourceStatus(
            spec.name,
            target,
            spec.required,
            False,
            False,
            spec.revision,
            None,
            spec.official_repository,
            None,
            None,
            "MISSING",
            f"source checkout not found: {target}",
        )
    if not (target / ".git").exists():
        return SourceStatus(
            spec.name,
            target,
            spec.required,
            True,
            False,
            spec.revision,
            None,
            spec.official_repository,
            None,
            None,
            "INVALID",
            f"existing path is not a Git checkout: {target}",
        )
    try:
        actual_commit = _git(target, "rev-parse", "HEAD").stdout.strip()
        actual_remote = _git(target, "remote", "get-url", "origin").stdout.strip()
        dirty = bool(_git(target, "status", "--porcelain").stdout.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        return SourceStatus(
            spec.name,
            target,
            spec.required,
            True,
            False,
            spec.revision,
            None,
            spec.official_repository,
            None,
            None,
            "INVALID",
            f"cannot inspect source checkout {target}: {exc}",
        )
    if dirty:
        reason = f"source checkout has local modifications; refusing to overwrite: {target}"
        status = "DIRTY"
    elif _normalize_remote(actual_remote) != _normalize_remote(spec.official_repository):
        reason = (
            f"source remote mismatch at {target}: expected {spec.official_repository}, "
            f"found {actual_remote}"
        )
        status = "WRONG_REMOTE"
    elif actual_commit != spec.revision:
        reason = (
            f"source commit mismatch at {target}: expected {spec.revision}, found {actual_commit}"
        )
        status = "WRONG_COMMIT"
    else:
        reason = "source checkout matches pinned remote and commit"
        status = "READY"
    return SourceStatus(
        spec.name,
        target,
        spec.required,
        True,
        status == "READY",
        spec.revision,
        actual_commit,
        spec.official_repository,
        actual_remote,
        dirty,
        status,
        reason,
    )


def fetch_source_checkout(
    spec: SourceSpec,
    external_root: Path,
    dry_run: bool = False,
) -> SourceStatus:
    existing = verify_source_checkout(spec, external_root)
    if not spec.required:
        return existing
    if existing.found:
        return existing
    target = external_root / spec.local_path
    if dry_run:
        return SourceStatus(
            spec.name,
            target,
            True,
            False,
            False,
            spec.revision,
            None,
            spec.official_repository,
            None,
            None,
            "WOULD_CLONE",
            f"would clone {spec.official_repository} at {spec.revision} into {target}",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                spec.official_repository,
                str(target),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        _git(target, "checkout", "--detach", spec.revision)
    except (OSError, subprocess.CalledProcessError) as exc:
        return SourceStatus(
            spec.name,
            target,
            True,
            target.exists(),
            False,
            spec.revision,
            None,
            spec.official_repository,
            None,
            None,
            "FETCH_FAILED",
            f"source fetch failed without deleting the checkout: {exc}",
        )
    return verify_source_checkout(spec, external_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and verify revision-pinned official model source checkouts."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="fetch all required sources")
    selection.add_argument("--source", choices=("paddleocr", "dbnet", "vietocr"))
    selection.add_argument("--verify", action="store_true", help="verify all required sources")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        specs = load_source_specs(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.source:
        selected = [specs[args.source]]
    else:
        selected = [spec for spec in specs.values() if spec.required]
    statuses = [
        (
            verify_source_checkout(spec, args.external_root)
            if args.verify
            else fetch_source_checkout(spec, args.external_root, args.dry_run)
        )
        for spec in selected
    ]
    for status in statuses:
        print(f"{status.status}: {status.name}: {status.reason}")
    if args.dry_run:
        return 0
    return 0 if all(status.ready for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())

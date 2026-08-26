"""Index canonical final GT by source documents without mutating GT files."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from invoice_ocr.io.paths import discover_documents, prediction_relative_path, sha256_file


class IndexModel(BaseModel):
    """Strict runtime-only index contract."""

    model_config = ConfigDict(extra="forbid")


class FinalGroundTruthTarget(IndexModel):
    document_id: str = Field(min_length=8)
    source_relative_path: str
    prediction_relative_path: str
    gt_relative_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FinalGroundTruthIndex(IndexModel):
    schema_version: str = "final-gt-index-v1"
    data_root: str
    gt_root: str
    final_root: str
    gt_prefix: str
    created_at: datetime
    target_count: int = Field(ge=0)
    excluded_gt_count: int = Field(ge=0)
    unmatched_source_count: int = Field(ge=0)
    documents: list[FinalGroundTruthTarget]
    excluded_gt_paths: list[str]
    unmatched_source_paths: list[str]

    @model_validator(mode="after")
    def validate_counts(self) -> FinalGroundTruthIndex:
        if self.target_count != len(self.documents):
            raise ValueError("target_count must equal the number of indexed documents")
        if self.excluded_gt_count != len(self.excluded_gt_paths):
            raise ValueError("excluded_gt_count must equal excluded_gt_paths length")
        if self.unmatched_source_count != len(self.unmatched_source_paths):
            raise ValueError("unmatched_source_count must equal unmatched_source_paths length")
        document_ids = [target.document_id for target in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("final GT index contains duplicate document IDs")
        predictions = [target.prediction_relative_path for target in self.documents]
        if len(predictions) != len(set(predictions)):
            raise ValueError("final GT index contains duplicate prediction paths")
        return self

    def by_document_id(self) -> dict[str, FinalGroundTruthTarget]:
        return {target.document_id: target for target in self.documents}


def canonical_final_root(gt_root: Path) -> Path:
    """Accept either the GT root or GT/final itself."""
    resolved = gt_root.expanduser().resolve()
    nested = resolved / "final"
    if resolved.name.casefold() == "final" or not nested.is_dir():
        return resolved
    return nested


def _safe_prefix(prefix: str | Path | None) -> Path | None:
    if prefix is None:
        return None
    candidate = Path(prefix)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("GT prefix must be a safe relative path")
    return Path() if str(candidate) in {"", "."} else candidate


def _choose_prefix(
    data_root: Path,
    final_root: Path,
    relative_predictions: list[Path],
    requested: Path | None,
) -> Path:
    if requested is not None:
        return requested
    candidates = [Path(), Path(data_root.expanduser().resolve().name)]
    scored: list[tuple[int, int, int, Path]] = []
    for order, prefix in enumerate(candidates):
        candidate_root = final_root / prefix
        filenames = Counter(path.name for path in candidate_root.rglob("*.json") if path.is_file())
        direct_count = sum(
            (candidate_root / relative).is_file() for relative in relative_predictions
        )
        total_count = sum(
            (candidate_root / relative).is_file() or filenames[relative.name] == 1
            for relative in relative_predictions
        )
        scored.append((direct_count, total_count, -order, prefix))
    best = max(scored, key=lambda score: score[:3])
    if best[1] == 0:
        checked = ", ".join(str(final_root / score[3]) for score in scored)
        raise FileNotFoundError(
            "no canonical final GT matched any source document; checked prefixes: " + checked
        )
    return best[3]


def build_final_ground_truth_index(
    data_root: Path,
    gt_root: Path,
    output_path: Path | None = None,
    gt_prefix: str | Path | None = None,
    *,
    require_all_sources: bool = False,
) -> FinalGroundTruthIndex:
    """Join source files to final JSON by relative path and record non-target JSON separately."""
    resolved_data = data_root.expanduser().resolve()
    resolved_gt = gt_root.expanduser().resolve()
    final_root = canonical_final_root(resolved_gt)
    if not final_root.is_dir():
        raise FileNotFoundError(f"canonical final GT directory not found: {final_root}")
    documents = discover_documents(resolved_data)
    relative_predictions = [
        prediction_relative_path(document.relative_path) for document in documents
    ]
    prefix = _choose_prefix(
        resolved_data,
        final_root,
        relative_predictions,
        _safe_prefix(gt_prefix),
    )
    selected_root = final_root / prefix
    targets: list[FinalGroundTruthTarget] = []
    unmatched: list[str] = []
    selected_paths: set[Path] = set()
    paths_by_name: dict[str, list[Path]] = {}
    for path in selected_root.rglob("*.json"):
        if path.is_file():
            paths_by_name.setdefault(path.name, []).append(path)
    for document, prediction_path in zip(documents, relative_predictions, strict=True):
        gt_path = selected_root / prediction_path
        if not gt_path.is_file():
            same_name = paths_by_name.get(prediction_path.name, [])
            if len(same_name) == 1:
                gt_path = same_name[0]
            else:
                unmatched.append(document.relative_path)
                continue
        resolved_path = gt_path.resolve()
        selected_paths.add(resolved_path)
        targets.append(
            FinalGroundTruthTarget(
                document_id=document.document_id,
                source_relative_path=document.relative_path,
                prediction_relative_path=prediction_path.as_posix(),
                gt_relative_path=resolved_path.relative_to(final_root).as_posix(),
                source_sha256=document.sha256,
                gt_sha256=sha256_file(resolved_path),
            )
        )
    if unmatched and require_all_sources:
        preview = ", ".join(unmatched[:5])
        suffix = "" if len(unmatched) <= 5 else f" and {len(unmatched) - 5} more"
        raise FileNotFoundError(
            f"canonical final GT is missing for {len(unmatched)} source documents under "
            f"{selected_root}: {preview}{suffix}"
        )
    all_gt_paths = sorted(
        path.resolve() for path in selected_root.rglob("*.json") if path.is_file()
    )
    excluded = [
        path.relative_to(final_root).as_posix()
        for path in all_gt_paths
        if path not in selected_paths
    ]
    index = FinalGroundTruthIndex(
        data_root=str(resolved_data),
        gt_root=str(resolved_gt),
        final_root=str(final_root),
        gt_prefix=prefix.as_posix() if prefix.parts else "",
        created_at=datetime.now(timezone.utc),
        target_count=len(targets),
        excluded_gt_count=len(excluded),
        unmatched_source_count=len(unmatched),
        documents=targets,
        excluded_gt_paths=excluded,
        unmatched_source_paths=unmatched,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(index.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return index


def load_final_ground_truth_index(path: Path) -> FinalGroundTruthIndex:
    if not path.is_file():
        raise FileNotFoundError(f"final GT target manifest not found: {path}")
    return FinalGroundTruthIndex.model_validate_json(path.read_text(encoding="utf-8"))


def final_gt_path(index: FinalGroundTruthIndex, target: FinalGroundTruthTarget) -> Path:
    path = Path(index.final_root) / Path(target.gt_relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"indexed canonical final GT no longer exists: {path}")
    if sha256_file(path) != target.gt_sha256:
        raise ValueError(f"indexed canonical final GT changed after indexing: {path}")
    return path

"""Geometry-based table cell grouping."""

from __future__ import annotations

from collections import defaultdict

from invoice_ocr.contracts import TableCell


def group_cells_by_row(cells: list[TableCell]) -> list[list[TableCell]]:
    grouped: dict[tuple[str, int], list[TableCell]] = defaultdict(list)
    for cell in cells:
        grouped[(cell.table_id, cell.row_index)].append(cell)
    return [
        sorted(row, key=lambda cell: cell.column_index)
        for _, row in sorted(grouped.items(), key=lambda item: item[0])
    ]


def validate_table_indices(cells: list[TableCell]) -> None:
    """Reject duplicate logical cell positions rather than silently overwriting."""
    positions: set[tuple[str, int, int]] = set()
    for cell in cells:
        position = (cell.table_id, cell.row_index, cell.column_index)
        if position in positions:
            raise ValueError(f"duplicate table cell position: {position}")
        positions.add(position)

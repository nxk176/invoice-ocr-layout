"""Convert labeled table cells into canonical medicine items."""

from __future__ import annotations

from invoice_ocr.contracts import InvoiceItem, MedicineRow, TableCell
from invoice_ocr.postprocessing.dates import normalize_date
from invoice_ocr.postprocessing.numbers import parse_vietnamese_number

STRING_LABELS = {
    "RAW_DESCRIPTION": "raw_description",
    "MEDICINE_NAME": "medicine_name",
    "STRENGTH": "strength",
    "MANUFACTURER": "manufacturer",
    "COUNTRY_OF_MANUFACTURE": "country_of_manufacture",
    "ITEM_BID_PACKAGE": "bid_package_name",
    "ITEM_CONTRACT_REFERENCE": "contract_reference",
    "LOT_NUMBER": "lot_number",
    "UNIT": "unit",
}
NUMBER_LABELS = {
    "QUANTITY": "quantity",
    "UNIT_PRICE": "unit_price",
    "LINE_AMOUNT": "line_amount",
    "ITEM_VAT_RATE": "vat_rate_percent",
    "VAT_AMOUNT": "vat_amount",
}


def reconstruct_medicine_item(row: MedicineRow, line_number: int) -> InvoiceItem:
    values: dict[str, object] = {"line_number": line_number}
    for cell in sorted(row.cells, key=lambda item: item.column_index):
        label = (cell.label or "").removeprefix("B-").removeprefix("I-")
        text = cell.text.strip()
        if not text:
            continue
        if label in STRING_LABELS:
            field = STRING_LABELS[label]
            existing = values.get(field)
            values[field] = f"{existing} {text}" if existing else text
        elif label in NUMBER_LABELS:
            values[NUMBER_LABELS[label]] = parse_vietnamese_number(text)
        elif label == "EXPIRY_DATE":
            values["expiry_date"] = normalize_date(text)
    if row.raw_description and "raw_description" not in values:
        values["raw_description"] = row.raw_description
    return InvoiceItem.model_validate(values)


def reconstruct_rows(cells: list[TableCell]) -> list[MedicineRow]:
    grouped: dict[tuple[str, int], list[TableCell]] = {}
    for cell in cells:
        grouped.setdefault((cell.table_id, cell.row_index), []).append(cell)
    rows: list[MedicineRow] = []
    for (table_id, row_index), row_cells in sorted(grouped.items()):
        first = row_cells[0]
        rows.append(
            MedicineRow(
                document_id=first.document_id,
                source_path=first.source_path,
                page_index=first.page_index,
                model_name=first.model_name,
                model_revision=first.model_revision,
                processing_status=first.processing_status,
                table_id=table_id,
                row_index=row_index,
                cells=sorted(row_cells, key=lambda item: item.column_index),
            )
        )
    return rows

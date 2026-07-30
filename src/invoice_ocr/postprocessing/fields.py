"""Map labeled entities into the canonical invoice contract."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from invoice_ocr.contracts import (
    Buyer,
    DeliveryUnitValue,
    InvoiceDocument,
    InvoiceHeader,
    InvoiceTotals,
    LabeledEntity,
    Party,
    ProvenancedWorkflowValue,
    ReceiverNameValue,
    ReviewableWorkflowValue,
    WorkflowFields,
)
from invoice_ocr.postprocessing.dates import normalize_date
from invoice_ocr.postprocessing.money import parse_money
from invoice_ocr.postprocessing.numbers import parse_vietnamese_number


def load_workflow_defaults(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"workflow defaults file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("workflow defaults must be a YAML mapping")
    return loaded


def _entity_values(entities: list[LabeledEntity]) -> dict[str, str]:
    grouped: dict[str, list[LabeledEntity]] = defaultdict(list)
    for entity in entities:
        label = entity.label.removeprefix("B-").removeprefix("I-")
        grouped[label].append(entity)
    values: dict[str, str] = {}
    for label, labeled in grouped.items():
        ordered = sorted(
            labeled, key=lambda item: (item.page_index, item.bbox.y_min, item.bbox.x_min)
        )
        values[label] = " ".join(item.text.strip() for item in ordered if item.text.strip())
    return values


def entities_to_invoice(
    page_number: int,
    entities: list[LabeledEntity],
    workflow_defaults: dict[str, Any] | None = None,
) -> InvoiceDocument:
    values = _entity_values(entities)
    defaults = workflow_defaults or {}
    status = values.get("STATUS")
    invoice_type = values.get("INVOICE_TYPE")
    status_default = defaults.get("status")
    invoice_type_default = defaults.get("invoice_type")
    delivery = values.get("DELIVERY_UNIT")
    receiver = values.get("RECEIVER_NAME")
    return InvoiceDocument(
        page_number=page_number,
        workflow_fields=WorkflowFields(
            status=ProvenancedWorkflowValue(
                value=status if status is not None else status_default,
                source=(
                    "invoice"
                    if status is not None
                    else "workflow_default"
                    if status_default is not None
                    else "not_found"
                ),
            ),
            invoice_type=ProvenancedWorkflowValue(
                value=invoice_type if invoice_type is not None else invoice_type_default,
                source=(
                    "invoice"
                    if invoice_type is not None
                    else "workflow_default"
                    if invoice_type_default is not None
                    else "not_found"
                ),
            ),
            bid_package=ReviewableWorkflowValue(
                value=values.get("BID_PACKAGE"),
                source="invoice" if values.get("BID_PACKAGE") else "not_found",
                needs_review=False,
            ),
            delivery_unit=DeliveryUnitValue(
                value=delivery,
                suggested_value=None,
                source="invoice" if delivery else "not_found",
                needs_review=False,
            ),
            receiver_name=ReceiverNameValue(
                value=receiver,
                candidate=None,
                source="invoice" if receiver else "not_found",
                needs_review=False,
            ),
        ),
        invoice=InvoiceHeader(
            invoice_number=values.get("INVOICE_NUMBER"),
            invoice_serial=values.get("INVOICE_SERIAL"),
            invoice_date=normalize_date(values.get("INVOICE_DATE")),
            payment_method=values.get("PAYMENT_METHOD"),
            invoice_lookup_code=values.get("INVOICE_LOOKUP_CODE"),
            contract_reference=values.get("CONTRACT_REFERENCE"),
        ),
        supplier=Party(
            supplier_name=values.get("SUPPLIER_NAME"),
            tax_code=values.get("SELLER_TAX_CODE"),
            address=values.get("SELLER_ADDRESS"),
            phone=values.get("SELLER_PHONE"),
        ),
        buyer=Buyer(
            buyer_contact=values.get("BUYER_CONTACT"),
            buyer_organization=values.get("BUYER_ORGANIZATION"),
            tax_code=values.get("BUYER_TAX_CODE"),
            budgetary_unit_code=values.get("BUYER_BUDGETARY_UNIT_CODE"),
            address=values.get("BUYER_ADDRESS"),
        ),
        totals=InvoiceTotals(
            subtotal_excluding_vat=parse_money(values.get("SUBTOTAL")),
            vat_rate_percent=parse_vietnamese_number(values.get("VAT_RATE")),
            vat_total=parse_money(values.get("VAT_TOTAL")),
            grand_total=parse_money(values.get("GRAND_TOTAL")),
            amount_in_words=values.get("AMOUNT_IN_WORDS"),
        ),
    )

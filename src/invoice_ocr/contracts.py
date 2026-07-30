"""Validated intermediate and canonical data contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base model that rejects accidental contract drift."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_AVAILABLE = "not_available"


class Point(StrictModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class BoundingBox(StrictModel):
    """Axis-aligned pixel bounding box using an exclusive max corner."""

    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(gt=0)
    y_max: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_extent(self) -> BoundingBox:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must have positive width and height")
        return self

    def normalize(self, width: int, height: int, scale: int = 1000) -> BoundingBox:
        """Return a box normalized to the LayoutLM coordinate range."""
        if width <= 0 or height <= 0:
            raise ValueError("page width and height must be positive")
        return BoundingBox(
            x_min=min(scale, max(0, self.x_min / width * scale)),
            y_min=min(scale, max(0, self.y_min / height * scale)),
            x_max=min(scale, max(0, self.x_max / width * scale)),
            y_max=min(scale, max(0, self.y_max / height * scale)),
        )


class StageRecord(StrictModel):
    document_id: str = Field(min_length=8)
    source_path: str = Field(min_length=1)
    page_index: int = Field(ge=0)
    model_name: str = Field(min_length=1)
    model_revision: str | None = None
    processing_status: ProcessingStatus


class SourceDocument(StrictModel):
    document_id: str = Field(min_length=8)
    source_path: str
    relative_path: str
    media_type: Literal["pdf", "image"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int | None = Field(default=None, ge=1)


class OrientationMetadata(StrictModel):
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    confidence: float | None = Field(default=None, ge=0, le=1)
    method: str = "none"


class DocumentPage(StageRecord):
    image_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    orientation: OrientationMetadata = Field(default_factory=OrientationMetadata)
    deskew_angle_degrees: float = 0.0


class DetectionRegion(StageRecord):
    region_id: str
    polygon: list[Point] = Field(min_length=4)
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, polygon: list[Point]) -> list[Point]:
        if len({(point.x, point.y) for point in polygon}) < 3:
            raise ValueError("polygon must contain at least three distinct points")
        return polygon


class RecognizedRegion(StageRecord):
    region_id: str
    polygon: list[Point] = Field(min_length=4)
    bbox: BoundingBox
    text: str
    confidence: float = Field(ge=0, le=1)


class LabeledEntity(StageRecord):
    entity_id: str
    label: str
    text: str
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)
    region_ids: list[str] = Field(default_factory=list)


class Relation(StageRecord):
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class TableCell(StageRecord):
    table_id: str
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    text: str
    bbox: BoundingBox
    label: str | None = None


class MedicineRow(StageRecord):
    table_id: str
    row_index: int = Field(ge=0)
    cells: list[TableCell] = Field(default_factory=list)
    raw_description: str | None = None


class InvoiceItem(StrictModel):
    line_number: int = Field(ge=1)
    is_promotion: bool = False
    raw_description: str | None = None
    medicine_name: str | None = None
    strength: str | None = None
    manufacturer: str | None = None
    country_of_manufacture: str | None = None
    bid_package_name: str | None = None
    contract_reference: str | None = None
    lot_number: str | None = None
    expiry_date: str | None = None
    unit: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    line_amount: Decimal | None = None
    vat_rate_percent: Decimal | None = None
    vat_amount: Decimal | None = None


class ProvenancedWorkflowValue(StrictModel):
    value: str | None = None
    source: Literal["invoice", "workflow_default", "not_found"] = "not_found"


class ReviewableWorkflowValue(StrictModel):
    value: str | None = None
    source: Literal["invoice", "not_found"] = "not_found"
    needs_review: bool = False


class DeliveryUnitValue(StrictModel):
    value: str | None = None
    suggested_value: str | None = None
    source: Literal["invoice", "supplier_name", "not_found"] = "not_found"
    needs_review: bool = False


class ReceiverNameValue(StrictModel):
    value: str | None = None
    candidate: str | None = None
    source: Literal["invoice", "buyer_signature_or_stamp", "not_found"] = "not_found"
    needs_review: bool = False


class WorkflowFields(StrictModel):
    status: ProvenancedWorkflowValue = Field(default_factory=ProvenancedWorkflowValue)
    invoice_type: ProvenancedWorkflowValue = Field(default_factory=ProvenancedWorkflowValue)
    bid_package: ReviewableWorkflowValue = Field(default_factory=ReviewableWorkflowValue)
    delivery_unit: DeliveryUnitValue = Field(default_factory=DeliveryUnitValue)
    receiver_name: ReceiverNameValue = Field(default_factory=ReceiverNameValue)


class InvoiceHeader(StrictModel):
    invoice_number: str | None = None
    invoice_serial: str | None = None
    invoice_date: str | None = None
    payment_method: str | None = None
    invoice_lookup_code: str | None = None
    contract_reference: str | None = None


class Party(StrictModel):
    supplier_name: str | None = None
    tax_code: str | None = None
    address: str | None = None
    phone: str | None = None


class Buyer(StrictModel):
    buyer_contact: str | None = None
    buyer_organization: str | None = None
    tax_code: str | None = None
    budgetary_unit_code: str | None = None
    address: str | None = None


class InvoiceTotals(StrictModel):
    subtotal_excluding_vat: Decimal | None = None
    vat_rate_percent: Decimal | None = None
    vat_total: Decimal | None = None
    grand_total: Decimal | None = None
    amount_in_words: str | None = None


class InvoiceValidation(StrictModel):
    subtotal_plus_vat_equals_grand_total: bool | None = None
    sum_of_items_equals_subtotal: bool | None = None
    rounding_difference_detected: bool = False
    item_count: int = Field(default=0, ge=0)
    unresolved_required_fields: list[str] = Field(default_factory=list)


class InvoiceDocument(StrictModel):
    page_number: int = Field(ge=1)
    workflow_fields: WorkflowFields = Field(default_factory=WorkflowFields)
    invoice: InvoiceHeader = Field(default_factory=InvoiceHeader)
    supplier: Party = Field(default_factory=Party)
    buyer: Buyer = Field(default_factory=Buyer)
    items: list[InvoiceItem] = Field(default_factory=list)
    totals: InvoiceTotals = Field(default_factory=InvoiceTotals)
    validation: InvoiceValidation = Field(default_factory=InvoiceValidation)


class InvoiceBatch(StrictModel):
    document_type: Literal["VAT_INVOICE_BATCH"] = "VAT_INVOICE_BATCH"
    invoice_count: int = Field(default=0, ge=0)
    invoices: list[InvoiceDocument] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_count(self) -> InvoiceBatch:
        if self.invoice_count != len(self.invoices):
            raise ValueError("invoice_count must equal the number of invoices")
        return self


class ModelIdentity(StrictModel):
    name: str
    revision: str | None = None
    checkpoint: str | None = None


class RunManifest(StrictModel):
    run_id: str
    command: str
    detector: ModelIdentity | None = None
    recognizer: ModelIdentity | None = None
    layout: ModelIdentity | None = None
    input_path: str | None = None
    ground_truth_path: str | None = None
    output_path: str
    device: str
    seed: int
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    document_count: int = 0
    failed_document_count: int = 0
    settings: dict[str, Any] = Field(default_factory=dict)


class DocumentError(StrictModel):
    document_id: str
    source_path: str
    page_index: int | None = None
    stage: str
    error_type: str
    message: str
    recoverable: bool = True
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def json_compatible(model: BaseModel) -> dict[str, Any]:
    """Serialize Decimal and datetime values with JSON-compatible Pydantic behavior."""
    return model.model_dump(mode="json", exclude_none=False)


def ensure_relative_path(path: Path) -> str:
    """Return a portable POSIX representation for persisted relative paths."""
    return path.as_posix()


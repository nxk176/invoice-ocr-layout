"""Domain-specific exceptions with actionable user messages."""


class InvoiceOCRError(Exception):
    """Base error for expected framework failures."""


class ConfigurationError(InvoiceOCRError):
    """Configuration is invalid or incomplete."""


class DependencyUnavailableError(InvoiceOCRError):
    """An optional backend dependency is not installed or importable."""


class CheckpointUnavailableError(InvoiceOCRError):
    """A required model checkpoint is missing or invalid."""


class AnnotationUnavailableError(InvoiceOCRError):
    """Ground-truth annotations required by an operation are absent."""


class NoInputDocumentsError(InvoiceOCRError):
    """No supported PDF or image documents were found."""


class OutputExistsError(InvoiceOCRError):
    """A valid output exists and --force was not supplied."""


class InvalidGroundTruthError(InvoiceOCRError):
    """Ground-truth files do not conform to the documented contract."""


"""DBNet++ adapter using the official DB repository."""

from invoice_ocr.adapters.detectors.dbnet import DBNetDetector


class DBNetPPDetector(DBNetDetector):
    name = "dbnetpp"
    algorithm = "DB++"

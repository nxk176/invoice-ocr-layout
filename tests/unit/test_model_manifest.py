from invoice_ocr.adapters.detectors.dbnet import DBNetDetector
from invoice_ocr.adapters.layout.layoutlmv3_complete import LayoutLMv3Adapter
from invoice_ocr.adapters.recognizers.vietocr import VietOCRRecognizer
from invoice_ocr.model_manifest import adapter_revision


def test_adapters_persist_exact_manifest_revision(tmp_path) -> None:
    assert DBNetDetector(tmp_path).revision == adapter_revision("detector", "dbnet")
    assert VietOCRRecognizer(tmp_path).revision == adapter_revision("recognizer", "vietocr")
    assert LayoutLMv3Adapter(tmp_path).revision == adapter_revision("layout", "layoutlmv3")
    assert len(DBNetDetector(tmp_path).revision or "") == 40

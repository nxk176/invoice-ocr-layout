# Ground-truth contract

Ground truth is private and ignored by Git. Paths mirror the source document where noted.

- `final/<relative_document_path_without_extension>.json`: canonical final JSON. This alone
  enables final extraction metrics.
- `detection/<document_id>.json`: page polygons/boxes for detector training and IoU metrics.
- `recognition/<document_id>.json`: region boxes plus exact transcriptions for recognizer
  training and CER/WER metrics.
- `layout/<document_id>.json`: OCR tokens, normalized token boxes, BIO labels, and optional
  entity relations for KIE fine-tuning and metrics.
- `tables/<document_id>.json`: table cells and medicine row membership.

Stage annotations are never inferred from final JSON. Missing detection annotations make
detection metrics N/A and block detector training. Missing transcriptions make recognition
metrics N/A and block recognizer training. Missing token boxes or labels block layout
fine-tuning. Run `python -m invoice_ocr.cli validate-gt --gt GT` for an exact report.

All annotations must use zero-based `page_index`; canonical invoice output uses one-based
`page_number`. Splits are deterministic and persisted as a manifest.


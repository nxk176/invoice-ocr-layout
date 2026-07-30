# Ground-truth contract

Ground truth is private and ignored by Git. Paths mirror the source document where noted.

- `final/<relative_document_path_without_extension>.json`: canonical final JSON. This alone
  enables final extraction metrics.
- `detection/<document_id>.json`: page/image references and text polygons or boxes.
- `recognition/<document_id>.json`: source page plus crop box (or a crop reference) and exact
  transcription.
- `layout/<document_id>.json`: OCR tokens, normalized token boxes, BIO labels, and optional
  entity/table relations.
- `tables/<document_id>.json`: table cells and medicine row membership.
- `splits/split_v1.json`: immutable document-level train/validation/test experiment manifest.

Create the locked split only after the dataset and the required annotations have been placed:

```bash
python -m invoice_ocr.cli create-split \
  --data data \
  --gt GT \
  --output GT/splits/split_v1.json \
  --seed 42
```

The manifest stores train, validation, and test document IDs; random seed; data/GT content
hashes; grouping rules; creation time; and a self-hash. Split files are excluded from the GT
content hash so writing the manifest does not invalidate itself. Pages from one source document
stay together. A changed data/GT hash requires a new split version; do not silently replace a
locked test set.

Stage annotations are never inferred from final JSON. Missing detection annotations make
detection metrics N/A and block detector training. Missing transcriptions make recognition
metrics N/A and block recognizer training. Missing token boxes or labels block layout
fine-tuning. Test annotations are never substituted for missing train or validation annotations.
Best checkpoint selection consumes validation metrics only.

Run `python -m invoice_ocr.cli validate-gt --gt GT` for an exact report. All annotations use
zero-based `page_index`; canonical invoice output uses one-based `page_number`.

# Third-party notices

This repository contains integration code only. It does not redistribute third-party
source repositories or model weights.

Supported upstream projects:

- PaddleOCR core OCR code — Apache License 2.0.
- DBNet / DBNet++ official MhLiao/DB repository — Apache License 2.0.
- VietOCR — Apache License 2.0.
- Microsoft UniLM repository — MIT at repository level; the LayoutLMv3 component and
  `microsoft/layoutlmv3-base` model card state CC-BY-NC-SA-4.0.
- PaddleOCR PP-Structure VI-LayoutXLM documentation/model assets state
  CC-BY-NC-SA-4.0; verify the exact artifact terms before use.
- Hugging Face Transformers — Apache License 2.0.

Model checkpoints can have additional dataset or usage terms. Review the `license`,
`official_repository`, `revision`, and `checkpoint_identifier` fields in
`configs/models/*.yaml` before downloading or deploying a checkpoint.

The downloader never treats a missing fine-tuned invoice checkpoint as available. Neither
the LayoutLMv3 base checkpoint nor the VI-LayoutXLM XFUND checkpoint is trained for this
repository's invoice labels.


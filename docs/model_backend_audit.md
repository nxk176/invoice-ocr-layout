# Audit backend model thật

Ngày audit: 2026-07-30. Audit này đánh giá production code path, không suy luận khả năng hỗ trợ
chỉ từ registry/class name. Máy local tại thời điểm audit không cài PaddlePaddle, PaddleOCR,
VietOCR, PyTorch, Transformers hoặc PaddleNLP; `models/` và `external/` cũng chưa có checkpoint
hay source checkout. Vì vậy trạng thái dưới đây phân biệt rõ “implementation có thật trong
code” với “đã load thành công trên server”.

| Backend | Stage | Inference thật | Training thật | Package/source | Pretrained weight | Revision pinned | Trạng thái | Việc còn thiếu |
|---|---|---:|---:|---|---|---|---|---|
| PaddleOCR detector | detector | Có, khi package/runtime/checkpoint hợp lệ | Không | `paddleocr==2.9.1`; source PaddleOCR cho training | `ch_PP-OCRv4_det_infer` | `07603421c20a96bb94bb87d0c4211032527ae706` (`v2.9.1`) | PARTIAL | Cần PaddlePaddle tương thích; training chưa có dataset conversion/config/validation candidate |
| DBNet | detector | Không | Không | Official `MhLiao/DB` checkout | Official trained-model collection, tải thủ công | `65ca77a0bcfbd7114b916cf8a1e9ca85114286ce` | SCAFFOLD/BLOCKED | Adapter chưa gọi model/representer; upstream yêu cầu stack PyTorch/CUDA cũ; chưa có validated training wrapper |
| DBNet++ | detector | Không | Không | Official `MhLiao/DB` checkout | Official DBNet++ Google Drive collection, tải thủ công | `65ca77a0bcfbd7114b916cf8a1e9ca85114286ce` | SCAFFOLD/BLOCKED | Giống DBNet; không được fallback sang PaddleOCR hoặc DBNet thường |
| PaddleOCR recognizer | recognizer | Có, khi package/runtime/checkpoint hợp lệ | Không | `paddleocr==2.9.1`; source PaddleOCR cho training | `latin_PP-OCRv3_rec_infer` | `07603421c20a96bb94bb87d0c4211032527ae706` (`v2.9.1`) | PARTIAL | Cần PaddlePaddle tương thích; training chưa có reviewed config/dataset/validation selection |
| VietOCR | recognizer | Có, khi package/PyTorch/checkpoint hợp lệ | Không | `vietocr==0.3.13`; không cần source checkout | `vgg_transformer` (`transformerocr.pth`) | `fe8c3a7fc714aec57ab81cec844eb3adf0c1636c` | PARTIAL | Existing trainer gọi CLI/config không tồn tại; phải tích hợp Trainer API, resume và validation-only selection |
| LayoutLMv3 | layout/KIE | Có, chỉ với invoice fine-tuned checkpoint | Có trong locked protocol | Hugging Face Transformers; không clone UniLM | `microsoft/layoutlmv3-base` encoder | UniLM `833df7e7832e5064a281131ee64a481afa8e5b95`; HF `cfbbbff0762e6aab37086fdd4739ad14fe7d5db4` | IMPLEMENTED, checkpoint-dependent | Base encoder không inference invoice; cần annotation, base snapshot và train để tạo `models/layoutlmv3/invoice-best` |
| VI-LayoutXLM | layout/KIE | Không | Không | Official PaddleOCR/PP-Structure source | `ser_vi_layoutxlm_xfund_pretrained` | `07603421c20a96bb94bb87d0c4211032527ae706` (`v2.9.1`) | SCAFFOLD | XFUND labels không tương thích invoice; adapter/trainer chưa gọi official SER runtime/config |

“Training thật” trong bảng chỉ được ghi Có nếu production code chuẩn bị dataset, chạy optimizer,
resume checkpoint, đánh giá validation và chọn best checkpoint không dùng test set. Vì lý do đó,
detector/recognizer dispatch hiện có vẫn được ghi Không.

## 1. Quyết định package hay source checkout

### PaddleOCR detector và recognizer

- Inference dùng package chính thức `paddleocr==2.9.1` và một PaddlePaddle CPU/GPU build do
  người vận hành chọn theo CUDA của server.
- Package đủ cho `PaddleOCR(...).ocr(...)`, nhưng không cung cấp source tree ổn định cho
  `tools/train.py`. Training cần `external/PaddleOCR/` tại commit
  `07603421c20a96bb94bb87d0c4211032527ae706`.
- Adapter tạo engine một lần, không reload theo page. Checkpoint local phải được resolve từ
  `models/paddleocr/detector` hoặc `models/paddleocr/recognizer`; không cho package âm thầm
  tải một checkpoint khác khi framework đã khai báo model.
- Inference detector trả polygon/bbox `DetectionRegion`; recognizer crop region và trả
  `RecognizedRegion`. Không có mock fallback.

### DBNet và DBNet++

- Official repository: <https://github.com/MhLiao/DB>.
- Không có stable pip package, vì vậy source checkout bắt buộc tại `external/DB/`.
- Exact commit hiện là HEAD upstream được audit:
  `65ca77a0bcfbd7114b916cf8a1e9ca85114286ce`.
- Upstream README yêu cầu PyTorch 1.2, CUDA 9/10.1 và custom deformable-convolution build.
  Đây không phải stack đã được xác minh tương thích với Python 3.10–3.12 của project.
- Current adapter chỉ kiểm tra dependency/checkpoint rồi chủ động báo runtime chưa tích hợp.
  Nó không load model, không forward ảnh và không chạy representer. Vì vậy inference/training
  đều không ready dù source/checkpoint có tồn tại.
- Official checkpoints được upstream phát hành qua Google Drive/Baidu Drive, không có direct
  artifact URL/checksum ổn định để downloader tự động dùng. Manifest ghi rõ manual setup,
  không tạo URL hoặc hash giả.

### VietOCR

- Inference dùng `vietocr==0.3.13`; package chứa `Cfg` và `Predictor`.
- `Predictor` load weight một lần; official translate path gọi `model.eval()` và
  `torch.no_grad()`.
- Source pin để provenance là
  `fe8c3a7fc714aec57ab81cec844eb3adf0c1636c`, nhưng checkout
  `external/vietocr/` không bắt buộc vì package chứa cả Predictor và Trainer API.
- Current project training path tìm một `vietocr` CLI và static config chưa tồn tại. Do đó
  training chưa được đánh dấu ready, dù upstream có Trainer API.

### LayoutLMv3

- Dùng `transformers==4.44.2`, PyTorch và Accelerate; không clone `microsoft/unilm`.
- Base snapshot là `microsoft/layoutlmv3-base` tại Hugging Face revision
  `cfbbbff0762e6aab37086fdd4739ad14fe7d5db4`.
- Base model chỉ là encoder. Nó không được báo cáo như pretrained invoice extractor với random
  head.
- Locked training tạo invoice token-classification head, hỗ trợ `linear_probe` và
  `full_finetune`, đánh giá mỗi epoch trên validation, dùng `load_best_model_at_end`, rồi copy
  validation-selected checkpoint vào `best/`.
- Inference adapter load processor/model một lần, gọi `model.eval()` và `torch.no_grad()`, dùng
  tokenizer `word_ids()` để map token prediction về `LabeledEntity`.

### VI-LayoutXLM

- Dùng official PaddleOCR/PP-Structure KIE checkout chung tại `external/PaddleOCR/`.
- Official config pin:
  `configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml`.
- Artifact XFUND là pretrained/generic SER checkpoint, không có invoice label mapping tương
  thích. Nó chỉ là training initialization, không phải invoice inference checkpoint.
- Current adapter/trainer chưa invoke official predictor/train/export commands và vẫn chủ động
  báo lỗi. Readiness luôn false cho đến khi dataset conversion, config override, output parser,
  resume và validation checkpoint selection được tích hợp/kiểm thử.

## 2. Checkpoint layout

| Model command | Local path | Loại |
|---|---|---|
| `paddleocr-detector` | `models/paddleocr/detector/` | official inference directory |
| `paddleocr-recognizer` | `models/paddleocr/recognizer/` | official inference directory |
| `dbnet` | `models/dbnet/totaltext_resnet50` | manual official checkpoint file |
| `dbnetpp` | `models/dbnetpp/td500_resnet50_deform_thre_asf` | manual official checkpoint file |
| `vietocr` | `models/vietocr/transformerocr.pth` | official pretrained file |
| `layoutlmv3-base` | `models/layoutlmv3-base/` | Hugging Face base snapshot |
| `vi-layoutxlm` | `models/vi_layoutxlm/base_xfund/` | official XFUND training checkpoint |

Invoice fine-tuned checkpoints không được downloader cung cấp. Layout inference mặc định cần
`models/layoutlmv3/invoice-best/`; VI-LayoutXLM sau khi hoàn thiện sẽ cần
`models/vi_layoutxlm/invoice-best/`.

## 3. Marker audit

Tìm toàn repository với `TODO|FIXME|pass|NotImplemented|mock|placeholder|scaffold|dummy|fake`
cho kết quả:

- Test-only hợp lệ: `MockDetector`, `MockRecognizer`, `MockLayout`,
  `MockProtocolExecutor`, fake CUDA/PDF/config classes trong `tests/`.
- Production `pass` hợp lệ: hai nhánh exception chỉ đánh dấu resource/schema không khả dụng,
  không phải model implementation.
- Production chưa hoàn thiện: DBNet/DBNet++ `detect()`, VI-LayoutXLM `extract()` và
  `train_vi_layoutxlm()` chủ động báo unavailable; PaddleOCR/VietOCR detector/recognizer
  trainers chưa có complete dataset/config/validation contract.
- Không có production backend fallback sang mock, dummy output hoặc model khác.

## 4. Readiness commands

```bash
python scripts/fetch_model_sources.py --all
python scripts/download_models.py --all
python -m invoice_ocr.cli verify-models \
  --all \
  --model-root models \
  --external-root external
```

`verify-models` mặc định thành công nếu mỗi backend được yêu cầu có ít nhất một capability thật
đang ready (`inference` hoặc `training`). Có thể kiểm tra chặt hơn bằng
`--require inference`, `--require training` hoặc `--require both`. Command không load model lớn
và không cần data; nó kiểm tra dependency, exact source commit, checkpoint/hash và audited
implementation capability.

Việc `verify-models` báo `NOT_READY` là kết quả audit hợp lệ, không phải lý do để tạo checkpoint
giả hay đổi sang backend khác.

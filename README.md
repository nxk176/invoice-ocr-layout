# invoice-ocr-layout

Framework OCR/KIE ghép module để fine-tune, inference và benchmark hóa đơn mua thuốc của
bệnh viện Việt Nam. Repository chỉ chứa code, schema, cấu hình và fixture tổng hợp; không
chứa hóa đơn, ground truth, checkpoint hay output thật.

## 1. Kiến trúc

Luồng xử lý cố định:

```text
PDF/image
  -> render từng page (page_index bắt đầu từ 0)
  -> sửa orientation EXIF
  -> deskew/enhancement
  -> text detection
  -> text recognition
  -> layout/KIE
  -> tái tạo table và medicine row
  -> normalize
  -> validation
  -> canonical JSON
```

Ba nhóm adapter độc lập:

- Detector: `paddleocr`, `dbnet`, `dbnetpp`.
- Recognizer: `paddleocr`, `vietocr`.
- Layout/KIE: `layoutlmv3`, `vi_layoutxlm`.

Thứ tự CLI `--detector A --recognizer B --layout C` luôn có nghĩa `A -> B -> C`. Có thể
viết gọn bằng `--pipeline A B C`; không được dùng đồng thời hai kiểu.

12 tổ hợp benchmark theo thứ tự deterministic:

1. `paddleocr -> paddleocr -> layoutlmv3`
2. `paddleocr -> paddleocr -> vi_layoutxlm`
3. `paddleocr -> vietocr -> layoutlmv3`
4. `paddleocr -> vietocr -> vi_layoutxlm`
5. `dbnet -> paddleocr -> layoutlmv3`
6. `dbnet -> paddleocr -> vi_layoutxlm`
7. `dbnet -> vietocr -> layoutlmv3`
8. `dbnet -> vietocr -> vi_layoutxlm`
9. `dbnetpp -> paddleocr -> layoutlmv3`
10. `dbnetpp -> paddleocr -> vi_layoutxlm`
11. `dbnetpp -> vietocr -> layoutlmv3`
12. `dbnetpp -> vietocr -> vi_layoutxlm`

Adapter thật không trả sample result. Nếu package hoặc checkpoint thiếu, command ghi lỗi theo
document vào `errors.jsonl`; mặc định document khác vẫn tiếp tục. `--fail-fast` dùng khi muốn
dừng ngay. Mock chỉ tồn tại trong test.

## 2. Cài đặt local Windows

Yêu cầu Python 3.10–3.12. Trong PowerShell:

```powershell
cd C:\Users\ADMIN\Desktop\rebuild\invoice-ocr-layout
python -m venv .venv
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1 `
  -WithPdf -WithPreprocessing -Dev
```

Backend GPU phải được cài theo matrix chính thức của PyTorch/PaddlePaddle phù hợp CUDA và
driver trên máy. Không cài ngẫu nhiên cả bản CPU lẫn GPU vào cùng environment.

PaddleOCR 2.9.1:

```powershell
python -m pip install "paddleocr==2.9.1"
# Sau đó cài đúng paddlepaddle hoặc paddlepaddle-gpu theo tài liệu chính thức.
```

VietOCR:

```powershell
python -m pip install "vietocr==0.3.13"
```

LayoutLMv3:

```powershell
python -m pip install -e ".[layoutlmv3]"
```

DBNet/DBNet++ không có API package ổn định. Clone repository chính thức ở ngoài repository
này, checkout commit ghi trong `configs/models/dbnet*.yaml`, cài dependency và đưa checkout
vào `PYTHONPATH`. Không copy source third-party vào đây. Adapter sẽ từ chối chạy nếu thiếu
runtime/checkpoint thay vì dùng một implementation không tương đương.

## 3. Cài đặt server Linux

Server không cần và không giả định có Codex:

```bash
cd /mnt/disk4/khainx/invoice-ocr-layout
conda activate nxk
bash scripts/setup_server.sh --with-pdf --with-preprocessing --dev
```

Script dùng environment hiện tại, có thể chạy lại, không tạo/xóa conda environment. Có thể
thêm `--with-paddleocr` hoặc `--with-layoutlmv3`; đọc kết quả kiểm tra CUDA sau cài đặt. Xem
mọi tùy chọn bằng:

```bash
bash scripts/setup_server.sh --help
bash scripts/run_server.sh --help
bash scripts/train_server.sh --help
bash scripts/benchmark_server.sh --help
```

## 4. Dữ liệu đầu vào

Mặc định:

```text
data/       PDF/image riêng tư
GT/         annotation riêng tư
models/     base weights và checkpoint fine-tuned
work/       rendered page và intermediate JSONL
outputs/    prediction, metrics, log và checkpoint training
```

Đặt PDF/image vào `data/`, có thể có thư mục con. ID document là SHA-256 deterministic của
relative path chuẩn hóa cộng content SHA-256, vì vậy hai file trùng tên ở hai thư mục không
bị đụng ID. Logic dùng `pathlib`, không hard-code đường dẫn Windows/Linux.

Nếu input trống, command kết thúc bằng thông báo `no input documents found`; không tạo
prediction giả và không hiện stack trace nội bộ.

## 5. Ground truth

Các mức annotation độc lập:

```text
GT/final/<relative_path_without_extension>.json
GT/detection/<document_id>.json
GT/recognition/<document_id>.json
GT/layout/<document_id>.json
GT/tables/<document_id>.json
```

- Chỉ có `final`: đánh giá final JSON được; detection IoU và recognition CER/WER là N/A.
- Train detector cần `pages[].regions` chứa boxes/polygons trong `GT/detection`.
- Train recognizer cần region và transcription string trong `GT/recognition`.
- Train layout cần token, box integer chuẩn hóa 0..1000 và BIO label trong `GT/layout`.
- Table/row annotation giữ membership cell/row; không suy ra annotation stage từ final JSON.

Kiểm tra GT:

```bash
python -m invoice_ocr.cli validate-gt --gt GT
```

Label dùng BIO. `configs/labels/invoice_bio_labels.yaml` là nguồn duy nhất của label map;
mọi semantic label được mở rộng thành `B-...` và `I-...`, ngoài ra có `O`. Table row được
tái tạo từ geometry/cell membership, không chỉ ghép danh sách token rời.

## 6. Tải base model

Weights không được commit:

```bash
python scripts/download_models.py --model paddleocr-detector
python scripts/download_models.py --model paddleocr-recognizer
python scripts/download_models.py --model vietocr
python scripts/download_models.py --model layoutlmv3-base
python scripts/download_models.py --model vi-layoutxlm-base
python scripts/download_models.py --all
```

Mỗi manifest trong `configs/models/` ghi official repository, exact revision, checkpoint
identifier/URL, local path, license, task và SHA-256 nếu upstream công bố. Khi upstream không
công bố SHA-256, downloader cảnh báo và lưu digest thực tải vào `DOWNLOAD.sha256`. Archive
được kiểm tra path traversal. DBNet/DBNet++ được báo là manual setup; `--all` không bịa URL.

Không có checkpoint fine-tuned hóa đơn trong downloader. `layoutlmv3-base` và
VI-LayoutXLM/XFUND chỉ là base/pretrained model.

## 7. Fine-tune

Pretrained LayoutLMv3 hoặc VI-LayoutXLM chưa biết các field hóa đơn của repository. Head
token-classification phải được tạo với label map invoice và fine-tune trên token boxes/labels
đã review. Base XFUND cũng không tự chuyển các nhãn form sang `INVOICE_NUMBER`,
`LOT_NUMBER`, `GRAND_TOTAL`, v.v.

Train từng stage:

```bash
python -m invoice_ocr.cli train \
  --stage detector \
  --model dbnetpp \
  --data data \
  --gt GT

python -m invoice_ocr.cli train \
  --stage recognizer \
  --model vietocr \
  --data data \
  --gt GT

python -m invoice_ocr.cli train \
  --stage layout \
  --model layoutlmv3 \
  --data data \
  --gt GT
```

Train tuần tự cả pipeline:

```bash
python -m invoice_ocr.cli train-pipeline \
  --pipeline paddleocr vietocr layoutlmv3 \
  --data data \
  --gt GT
```

Split train/validation/test dùng SHA-256 theo `document_id` và `--seed`, persist vào
`split_manifest.json`. `--resume` dùng checkpoint/split có sẵn; khác seed bị từ chối.
Transformers Trainer lưu checkpoint và metadata processor. PaddleOCR, VI-LayoutXLM và DBNet
phải dùng exact official checkout/config tương ứng manifest vì API train không ổn định.

Nếu annotation thiếu, lỗi nêu đúng folder/key cần bổ sung; final JSON không được biến thành
box/transcription/token label tự động.

## 8. Inference

Explicit:

```bash
python -m invoice_ocr.cli run \
  --detector paddleocr \
  --recognizer vietocr \
  --layout layoutlmv3 \
  --input data \
  --output outputs/run_001
```

Shortcut tương đương:

```bash
python -m invoice_ocr.cli run \
  --pipeline paddleocr vietocr layoutlmv3 \
  --input data \
  --output outputs/run_001
```

Các option chung có trên mọi subcommand:

```text
--config --device auto|cpu|cuda --batch-size --num-workers --resume --force
--seed --fail-fast --keep-intermediate --workflow-defaults --model-root --work-root
```

Prediction hợp lệ sẵn có không bị ghi đè nếu thiếu `--force`. `--resume` tái sử dụng artifact
hợp lệ; `--force` chỉ thay artifact của run được chỉ định. Mặc định một invoice page tạo một
invoice object, page output dùng số bắt đầu từ 1.

Chạy từng stage:

```bash
python -m invoice_ocr.cli detect \
  --detector paddleocr --input data --output work/stage_001
python -m invoice_ocr.cli recognize \
  --recognizer vietocr --input work/stage_001 --output work/stage_001
python -m invoice_ocr.cli extract \
  --layout layoutlmv3 --input work/stage_001 --output work/stage_001
python -m invoice_ocr.cli postprocess \
  --input work/stage_001 --output outputs/stage_001
python -m invoice_ocr.cli evaluate \
  --input outputs/stage_001/predictions --gt GT \
  --output outputs/stage_001/metrics.json
```

## 9. Benchmark

```bash
python -m invoice_ocr.cli benchmark \
  --all-combinations \
  --input data \
  --gt GT \
  --output outputs/benchmark_001
```

Benchmark tạo đúng 12 run, cô lập lỗi backend, rồi xuất:

- `metrics.json`, `metrics.csv`, `comparison.csv`, `summary.md`;
- detection IoU/precision/recall/F1 nếu có detection GT;
- recognition CER/WER/exact/numeric accuracy nếu có recognition GT;
- entity/relation/field metrics nếu có layout GT;
- field/normalized/medicine-row/document metrics nếu có final GT;
- runtime và failed-document count.

Metric thiếu annotation có `status: N/A` và `reason`, không được tính từ final GT.

## 10. Output contract

```text
work/<run_id>/pages.jsonl
work/<run_id>/detections.jsonl
work/<run_id>/recognitions.jsonl
work/<run_id>/entities.jsonl
work/<run_id>/tables.jsonl
outputs/<run_id>/predictions/*.json
outputs/<run_id>/metrics.json
outputs/<run_id>/manifest.json
outputs/<run_id>/summary.md
outputs/<run_id>/errors.jsonl
outputs/<run_id>/logs/run.log
```

Intermediate record luôn có `document_id`, `source_path`, zero-based `page_index`, model
name/revision và processing status. Canonical JSON validate bằng
`configs/schema/invoice.schema.json`; file JSON không chứa prose.

Ngày được chuẩn hóa `YYYY-MM-DD`; tiền/số lượng là JSON number; identifier như invoice
number, serial, lot/tax/contract/lookup code luôn là string để giữ leading zero. Field không
rõ là `null`. Parser không biến `NSX: Việt Nam` thành manufacturer, không coi contract là bid
package, không suy delivery unit từ supplier, và tên gần chữ ký chỉ là candidate cần review.

Validation chỉ ghi kết quả:

```text
subtotal_excluding_vat + vat_total ≈ grand_total
sum(items.line_amount) ≈ subtotal_excluding_vat
```

Tolerance nằm trong workflow defaults. Không sửa số OCR theo phép tính và không bắt buộc
`quantity × displayed unit_price = line_amount`.

## 11. Lệnh server chuẩn

Sau `git pull`, data và GT phải nằm đúng:

```text
/mnt/disk4/khainx/invoice-ocr-layout/data
/mnt/disk4/khainx/invoice-ocr-layout/GT
```

Inference:

```bash
cd /mnt/disk4/khainx/invoice-ocr-layout
conda activate nxk
bash scripts/run_server.sh \
  --pipeline paddleocr vietocr layoutlmv3 \
  --input /mnt/disk4/khainx/invoice-ocr-layout/data \
  --gt /mnt/disk4/khainx/invoice-ocr-layout/GT \
  --output /mnt/disk4/khainx/invoice-ocr-layout/outputs/run_001
```

Training:

```bash
bash scripts/train_server.sh \
  --stage layout \
  --model layoutlmv3 \
  --data /mnt/disk4/khainx/invoice-ocr-layout/data \
  --gt /mnt/disk4/khainx/invoice-ocr-layout/GT
```

Benchmark:

```bash
bash scripts/benchmark_server.sh \
  --all-combinations \
  --data /mnt/disk4/khainx/invoice-ocr-layout/data \
  --gt /mnt/disk4/khainx/invoice-ocr-layout/GT \
  --output /mnt/disk4/khainx/invoice-ocr-layout/outputs/benchmark_001
```

## 12. Tải result về local

Không hard-code host/password:

```powershell
.\scripts\download_results.ps1 `
  -ServerHost gpu.example.org `
  -ServerUser khainx `
  -RemoteRunDirectory /mnt/disk4/khainx/invoice-ocr-layout/outputs/run_001 `
  -LocalOutputDirectory .\outputs\run_001
```

Script ưu tiên `rsync`, fallback `scp`, dùng SSH key/agent do người vận hành cấu hình.

## 13. Kiểm thử

```bash
python -m ruff format src tests scripts
python -m ruff check src tests scripts
python -m mypy src/invoice_ocr
python -m pytest
python -m pytest tests/integration/test_mock_pipeline.py
```

Fixture là ảnh tổng hợp với tên/tổ chức hư cấu, không phải hóa đơn redacted từ dữ liệu thật.
Test data-safety kiểm tra extension nhạy cảm/binary lớn không bị Git track.

## 14. An toàn public repository

Tuyệt đối không push:

- hóa đơn PDF/image, crop/rendered page;
- GT thật, OCR output, prediction, metrics/log từ dữ liệu thật;
- `.env`, secret, host/password;
- `.pth`, `.pt`, `.bin`, `.onnx`, `.pdparams`, `.safetensors` hay checkpoint khác;
- cache Python/tool.

`.gitignore` chặn nội dung `data/`, `GT/`, `models/`, `work/`, `outputs/` và chỉ cho phép
README hướng dẫn ở bốn folder public.

## 15. Third-party và giới hạn

Xem exact revision/license trong `configs/models/*.yaml` và
`THIRD_PARTY_NOTICES.md`. Repository không vendor code/weights.

Giới hạn hiện tại:

- Chưa thể xác nhận độ chính xác hay runtime trên hóa đơn bệnh viện vì `data/` và `GT/` trống.
- Checkpoint LayoutLMv3/VI-LayoutXLM invoice chưa tồn tại cho đến khi fine-tune.
- DBNet/DBNet++ và PP-Structure KIE phụ thuộc official checkout/config revision-sensitive.
- Orientation ngoài EXIF cần một classifier được cấu hình riêng; không tự đoán rotation.
- Tái tạo row hình học cần được benchmark và điều chỉnh tolerance bằng GT tables thật.
- License LayoutLMv3/VI-LayoutXLM có điều kiện phi thương mại; cần legal review trước deploy.

## 16. Đánh giá pretrained và sau fine-tuning

Protocol `pretrained-vs-finetuned` trả lời một câu hỏi có kiểm soát: chất lượng và chi phí
inference thay đổi thế nào khi model được fine-tune trên hóa đơn thuốc. Pretrained baseline
của detector/recognizer là official checkpoint chạy trực tiếp trên test set khóa, không cập
nhật weight bằng dữ liệu invoice. Sau đó model được khởi tạo lại từ đúng pretrained checkpoint,
chỉ học trên train, chọn best checkpoint bằng validation, rồi chạy lại trên đúng test IDs.

LayoutLMv3 và VI-LayoutXLM base chỉ là encoder pretrained; chúng chưa biết invoice labels và
không trực tiếp sinh canonical invoice JSON. Framework không báo cáo encoder kèm random task
head như pretrained invoice extractor. So sánh chính là:

- `linear_probe`: khóa toàn bộ encoder, chỉ học invoice token-classification head trên train;
- `full_finetune`: học task head và cho phép encoder nhận gradient;
- `generic_kie_checkpoint`: chỉ dùng khi manifest chỉ ra checkpoint chính thức, revision,
  license và label space tương thích. Nếu không tương thích, kết quả là N/A, không ánh xạ tùy ý.

### 16.1 Tạo locked split

Đặt data và annotation thật vào các thư mục ignored trước, rồi tạo đúng một manifest version:

```bash
python -m invoice_ocr.cli create-split \
  --data data \
  --gt GT \
  --output GT/splits/split_v1.json \
  --seed 42
```

Manifest chứa `train_document_ids`, `validation_document_ids`, `test_document_ids`, seed,
data/GT hashes, grouping rules, creation time và self-hash. Mọi page của một source document
luôn ở cùng partition. Nếu data hoặc GT thay đổi, lệnh experiment từ chối dùng manifest cũ;
hãy tạo split version mới thay vì âm thầm thay test set.

Test IDs không được truyền vào training backend, early stopping, checkpoint selection hay
threshold tuning. Best checkpoint candidate bắt buộc ghi `evaluated_split=validation`; selector
từ chối candidate lấy từ train hoặc test. Hai evaluation giữ cùng preprocessing, decoding,
post-processing, schema, workflow defaults, tolerance, batch/workers và metric implementation.

### 16.2 Baseline, fine-tune và đánh giá lại

Ví dụ recognizer VietOCR:

```bash
python -m invoice_ocr.cli evaluate-model \
  --stage recognizer \
  --model vietocr \
  --checkpoint-source pretrained \
  --data data \
  --gt GT \
  --split-manifest GT/splits/split_v1.json \
  --split test \
  --output outputs/baselines/vietocr_pretrained

python -m invoice_ocr.cli train \
  --stage recognizer \
  --model vietocr \
  --checkpoint-source pretrained \
  --data data \
  --gt GT \
  --split-manifest GT/splits/split_v1.json \
  --output models/finetuned/vietocr_invoice

python -m invoice_ocr.cli evaluate-model \
  --stage recognizer \
  --model vietocr \
  --checkpoint models/finetuned/vietocr_invoice/best \
  --checkpoint-source finetuned \
  --data data \
  --gt GT \
  --split-manifest GT/splits/split_v1.json \
  --split test \
  --output outputs/finetuned/vietocr_invoice

python -m invoice_ocr.cli compare-runs \
  --before outputs/baselines/vietocr_pretrained \
  --after outputs/finetuned/vietocr_invoice \
  --output outputs/comparisons/vietocr
```

Detector dùng quy trình tương tự. Layout full fine-tune:

```bash
python -m invoice_ocr.cli train \
  --stage layout \
  --model layoutlmv3 \
  --layout-training-mode full_finetune \
  --checkpoint-source pretrained \
  --data data \
  --gt GT \
  --split-manifest GT/splits/split_v1.json \
  --output models/finetuned/layoutlmv3_invoice
```

### 16.3 Chạy trọn protocol

Một pipeline, theo đúng thứ tự detector → recognizer → layout:

```bash
python -m invoice_ocr.cli experiment \
  --pipeline paddleocr vietocr layoutlmv3 \
  --protocol pretrained-vs-finetuned \
  --layout-baseline-mode linear_probe \
  --layout-finetuned-mode full_finetune \
  --data data \
  --gt GT \
  --split-manifest GT/splits/split_v1.json \
  --output outputs/experiments/paddle_vietocr_layoutlmv3 \
  --resume
```

Toàn bộ 12 tổ hợp:

```bash
python -m invoice_ocr.cli experiment \
  --all-combinations \
  --protocol pretrained-vs-finetuned \
  --layout-baseline-mode linear_probe \
  --layout-finetuned-mode full_finetune \
  --data data \
  --gt GT \
  --split-manifest GT/splits/split_v1.json \
  --output outputs/experiments/all_combinations \
  --resume
```

`--resume` bỏ qua stage có manifest `success` và artifact hợp lệ; `--force` mới cho phép chạy
lại. Một model thiếu dependency/checkpoint/annotation được ghi `SKIPPED` kèm reason và không
làm dừng model khác. Không backend production nào tự chuyển sang mock.

### 16.4 Annotation bắt buộc

- Detector fine-tune/evaluate cần page/image reference và text polygon hoặc bounding box tại
  `GT/detection/<document_id>.json`.
- Recognizer cần crop, hoặc source page + crop box, và exact transcription tại
  `GT/recognition/<document_id>.json`.
- Layout cần OCR tokens/words, boxes, BIO entity labels và relation/table annotation khi metric
  tương ứng được yêu cầu tại `GT/layout/<document_id>.json`.
- End-to-end canonical metrics cần `GT/final/<relative_path>.json`.

Thiếu annotation làm stage training `SKIPPED` và metric tương ứng N/A có reason; final JSON
không được tự chuyển thành stage annotation.

### 16.5 Đọc artifact và comparison

Mỗi experiment có `pretrained/`, `training/`, `finetuned/` và `comparison/`. Training timing
được lưu riêng và không cộng vào inference timing. `timing.json` của mỗi evaluation ghi model
load, preprocessing, stage inference, reconstruction/post-processing/validation/evaluation,
wall time, throughput, document/page counts, CPU RAM và GPU memory. Resource không đo được là
`null` kèm reason. CUDA timing gọi synchronize trước/sau vùng đo; warm-up iterations không tính
vào inference chính.

Mỗi aggregate metric có value, numerator, denominator, evaluated/skipped counts, hướng
`lower_is_better` và N/A reason. Trong `comparison.json`, `metrics_before`/`metrics_after` là
chất lượng test; `runtime_before`/`runtime_after` là runtime/resource. CER, WER, latency và
memory thấp hơn là tốt hơn; precision/recall/F1/exact/accuracy/throughput cao hơn là tốt hơn.
`per_document.csv`, `per_field.csv`, `timing_comparison.csv` và `summary.md` giải thích phần
cải thiện/giảm. Không có test GT hợp lệ thì summary không kết luận fine-tuning tốt hơn.

Comparison mặc định từ chối khác split hash, test IDs hoặc schema version. Khác batch size,
device/hardware/config vẫn được ghi rõ là fairness difference. Chỉ dùng
`--allow-incomparable-runs` cho diagnostic được đánh dấu không công bằng.

### 16.6 Lệnh chính xác trên server

```bash
cd /mnt/disk4/khainx/invoice-ocr-layout
conda activate nxk
git pull

bash scripts/setup_server.sh

bash scripts/run_experiment_server.sh \
  --pipeline paddleocr vietocr layoutlmv3 \
  --protocol pretrained-vs-finetuned \
  --data /mnt/disk4/khainx/invoice-ocr-layout/data \
  --gt /mnt/disk4/khainx/invoice-ocr-layout/GT \
  --output /mnt/disk4/khainx/invoice-ocr-layout/outputs/experiments/run_001
```

Script dùng conda environment đang active, tự dùng/tạo mặc định
`GT/splits/split_v1.json`, hỗ trợ `--resume`/`--force`, và trả non-zero nếu toàn bộ experiment
đều thất bại hoặc bị skip. Dùng `--all-combinations` thay `--pipeline A B C` để chạy đủ 12 tổ
hợp. Server không cần và không được giả định có Codex.

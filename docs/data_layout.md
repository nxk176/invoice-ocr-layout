# Bố trí dữ liệu và artifact runtime

GitHub chỉ chứa source code, config, scripts, tests và documentation. Các thư mục dưới đây
chỉ tồn tại trên máy local hoặc server và tuyệt đối không được push:

| Thư mục | Nội dung |
| --- | --- |
| `data/` | PDF hoặc ảnh đầu vào riêng tư; hỗ trợ thư mục con. |
| `GT/` | Annotation từng stage và canonical JSON ground truth. |
| `models/` | Pretrained checkpoint và checkpoint đã fine-tune. |
| `work/` | Trang render, crop và intermediate JSONL. |
| `outputs/` | Predictions, metrics, logs và kết quả experiment. |
| `external/` | Official source checkout thủ công khi package Python không đủ ổn định. |

Các setup script và Python CLI tự tạo các thư mục này bằng thao tác idempotent. Chúng không
cần tồn tại ngay sau khi clone. Không commit hóa đơn, ground truth thật, weights, checkpoints,
intermediate artifacts hoặc kết quả OCR.

## Cấu trúc ground truth

```text
GT/
  final/<relative_document_path_without_extension>.json
  detection/<document_id>.json
  recognition/<document_id>.json
  layout/<document_id>.json
  tables/<document_id>.json
  splits/split_v1.json
```

- Detector cần image/page reference và text polygons hoặc bounding boxes trong
  `GT/detection/<document_id>.json`.
- Recognizer cần exact transcription cùng crop hoặc source page + crop box trong
  `GT/recognition/<document_id>.json`.
- Layout/KIE cần OCR words/tokens, bounding boxes và entity labels trong
  `GT/layout/<document_id>.json`; thêm row/table relations khi task yêu cầu.
- Final JSON evaluation cần canonical invoice JSON trong
  `GT/final/<relative_document_path_without_extension>.json`.
- `GT/tables/` lưu table cells và medicine-row membership.

Stage annotation không được suy ra từ final JSON. Nếu chỉ có `GT/final/`, framework vẫn đánh
giá final extraction nhưng detection IoU và recognition CER/WER là N/A. Thiếu annotation của
stage nào thì training stage đó bị chặn với lý do cụ thể. Mọi `page_index` trong annotation là
zero-based; `page_number` trong canonical output là one-based.

Tạo locked split sau khi đã đặt data và annotation cần thiết:

```bash
python -m invoice_ocr.cli create-split \
  --data data \
  --gt GT \
  --output GT/splits/split_v1.json \
  --seed 42
```

Manifest giữ train/validation/test document IDs, seed, hash dữ liệu/GT, grouping rules, thời
gian tạo và self-hash. Các trang của cùng source document luôn ở cùng split. Không thay thế
test set đã khóa; best checkpoint chỉ được chọn bằng validation set.

Kiểm tra contract GT:

```bash
python -m invoice_ocr.cli validate-gt --gt GT
```

## Đường dẫn mặc định

Local Windows:

```text
C:\Users\ADMIN\Desktop\rebuild\invoice-ocr-layout\data
C:\Users\ADMIN\Desktop\rebuild\invoice-ocr-layout\GT
C:\Users\ADMIN\Desktop\rebuild\invoice-ocr-layout\models
C:\Users\ADMIN\Desktop\rebuild\invoice-ocr-layout\work
C:\Users\ADMIN\Desktop\rebuild\invoice-ocr-layout\outputs
C:\Users\ADMIN\Desktop\rebuild\invoice-ocr-layout\external
```

Server Linux:

```text
/mnt/disk4/khainx/invoice-ocr-layout/data
/mnt/disk4/khainx/invoice-ocr-layout/GT
/mnt/disk4/khainx/invoice-ocr-layout/models
/mnt/disk4/khainx/invoice-ocr-layout/work
/mnt/disk4/khainx/invoice-ocr-layout/outputs
/mnt/disk4/khainx/invoice-ocr-layout/external
```

Không hard-code các đường dẫn tuyệt đối này trong logic. CLI dùng `pathlib` và nhận đường dẫn
qua option/config để cùng code chạy trên Windows và Linux.

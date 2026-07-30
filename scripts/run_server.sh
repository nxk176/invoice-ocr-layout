#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_server.sh \
    --pipeline DETECTOR RECOGNIZER LAYOUT \
    --input DATA_DIR \
    [--gt GT_DIR] \
    --output OUTPUT_DIR \
    [invoice_ocr run options]

--gt is validated when supplied; inference never reads GT to create predictions.
EOF
}

if [[ $# -eq 0 ]]; then usage; exit 2; fi
run_args=()
gt_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gt)
      [[ $# -ge 2 ]] || { echo "--gt requires a path" >&2; exit 2; }
      gt_dir="$2"
      shift 2
      ;;
    *)
      run_args+=("$1")
      shift
      ;;
  esac
done
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p \
  "$project_root/data" \
  "$project_root/GT" \
  "$project_root/models" \
  "$project_root/work" \
  "$project_root/outputs" \
  "$project_root/external"
if [[ -n "$gt_dir" ]]; then
  python -m invoice_ocr.cli validate-gt --gt "$gt_dir"
fi
python -m invoice_ocr.cli run "${run_args[@]}"


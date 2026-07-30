#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Run the locked pretrained-versus-fine-tuned protocol in the currently active environment.

Usage:
  bash scripts/run_experiment_server.sh \
    --pipeline paddleocr vietocr layoutlmv3 \
    --protocol pretrained-vs-finetuned \
    --data /mnt/disk4/khainx/invoice-ocr-layout/data \
    --gt /mnt/disk4/khainx/invoice-ocr-layout/GT \
    --output /mnt/disk4/khainx/invoice-ocr-layout/outputs/experiments/run_001

  bash scripts/run_experiment_server.sh \
    --all-combinations \
    --protocol pretrained-vs-finetuned \
    --data /mnt/disk4/khainx/invoice-ocr-layout/data \
    --gt /mnt/disk4/khainx/invoice-ocr-layout/GT \
    --output /mnt/disk4/khainx/invoice-ocr-layout/outputs/experiments/all_models

All additional options are forwarded to `python -m invoice_ocr.cli experiment`.
Use --resume to reuse completed, validated stages and --force to intentionally rerun.
The command returns non-zero when every requested experiment fails or is skipped.
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

for argument in "$@"; do
  if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
    usage
    exit 0
  fi
done

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is not available in the currently active conda environment." >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p \
  "$project_root/data" \
  "$project_root/GT" \
  "$project_root/models" \
  "$project_root/work" \
  "$project_root/outputs" \
  "$project_root/external"

python -m invoice_ocr.cli experiment "$@"
status=$?
if [[ $status -ne 0 ]]; then
  echo "Experiment did not produce any successful comparison; inspect experiment_summary.json." >&2
fi
exit "$status"

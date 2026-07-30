#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_server.sh \
    --stage detector|recognizer|layout \
    --model MODEL \
    --data DATA_DIR \
    --gt GT_DIR \
    [invoice_ocr train options]
EOF
}

if [[ $# -eq 0 ]]; then usage; exit 2; fi
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p \
  "$project_root/data" \
  "$project_root/GT" \
  "$project_root/models" \
  "$project_root/work" \
  "$project_root/outputs" \
  "$project_root/external"
python -m invoice_ocr.cli train "$@"


#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/benchmark_server.sh \
    --all-combinations \
    --data DATA_DIR \
    --gt GT_DIR \
    --output OUTPUT_DIR \
    [invoice_ocr benchmark options]
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
python -m invoice_ocr.cli benchmark "$@"


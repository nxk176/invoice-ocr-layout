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
python -m invoice_ocr.cli benchmark "$@"


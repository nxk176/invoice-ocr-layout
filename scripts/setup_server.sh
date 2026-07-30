#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_server.sh [options]

Install the project into the currently active environment. The script never creates,
removes, or replaces a conda environment.

Options:
  --with-pdf            Install pypdfium2 support.
  --with-preprocessing  Install OpenCV preprocessing support.
  --with-paddleocr      Install PaddleOCR package (PaddlePaddle runtime remains explicit).
  --with-layoutlmv3     Install Transformers and PyTorch dependencies.
  --dev                 Install development/test dependencies.
  -h, --help            Show this help.
EOF
}

extras=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-pdf) extras+=("pdf") ;;
    --with-preprocessing) extras+=("preprocessing") ;;
    --with-paddleocr) extras+=("paddleocr") ;;
    --with-layoutlmv3) extras+=("layoutlmv3") ;;
    --dev) extras+=("dev") ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p \
  "$project_root/data" \
  "$project_root/GT" \
  "$project_root/models" \
  "$project_root/work" \
  "$project_root/outputs" \
  "$project_root/external"

command -v python >/dev/null 2>&1 || {
  echo "Python is not available in the current environment." >&2
  exit 2
}
python - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is unsupported; use Python 3.10, 3.11, or 3.12."
    )
print(f"Using Python {sys.version.split()[0]} at {sys.executable}")
PY

specifier="."
if [[ ${#extras[@]} -gt 0 ]]; then
  joined="$(IFS=,; echo "${extras[*]}")"
  specifier=".[$joined]"
fi
python -m pip install -e "$specifier"

python - <<'PY'
import importlib.util

torch_spec = importlib.util.find_spec("torch")
paddle_spec = importlib.util.find_spec("paddle")
if torch_spec:
    import torch
    print(f"PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}")
else:
    print("PyTorch is not installed; LayoutLMv3 commands will report a dependency error.")
if paddle_spec:
    import paddle
    print(f"PaddlePaddle {paddle.__version__}; compiled_with_cuda={paddle.is_compiled_with_cuda()}")
else:
    print("PaddlePaddle is not installed; PaddleOCR/VI-LayoutXLM need an explicit CPU/CUDA build.")
PY

echo "Setup completed without changing the active environment."
echo "Check official PaddlePaddle/PyTorch CUDA matrices before installing GPU runtimes."


#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_server.sh [options]

Install into the currently active environment. The script never creates, removes,
or replaces a conda environment.

Model selection (choose at most one):
  --minimal                         Install only the project (default).
  --pipeline DETECTOR RECOGNIZER LAYOUT
                                    Install/fetch/download only one A -> B -> C pipeline.
  --all-models                      Prepare every declared backend.

Additional options:
  --with-pdf            Install pypdfium2 support.
  --with-preprocessing  Install OpenCV preprocessing support.
  --with-paddleocr      Install the PaddleOCR Python package.
  --with-layoutlmv3     Install Transformers/PyTorch dependencies.
  --dev                 Install development/test dependencies.
  -h, --help            Show this help.

PaddlePaddle itself is hardware-specific and must be installed from PaddlePaddle's
official CPU/CUDA matrix before a Paddle backend can pass verification.
EOF
}

mode="minimal"
mode_selected=false
pipeline=()
extras=()

select_mode() {
  local requested="$1"
  if [[ "$mode_selected" == true ]]; then
    echo "Choose only one of --minimal, --pipeline, or --all-models." >&2
    exit 2
  fi
  mode="$requested"
  mode_selected=true
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --minimal)
      select_mode "minimal"
      shift
      ;;
    --pipeline)
      select_mode "pipeline"
      if [[ $# -lt 4 ]]; then
        echo "--pipeline requires DETECTOR RECOGNIZER LAYOUT." >&2
        exit 2
      fi
      pipeline=("$2" "$3" "$4")
      shift 4
      ;;
    --all-models)
      select_mode "all"
      shift
      ;;
    --with-pdf)
      extras+=("pdf")
      shift
      ;;
    --with-preprocessing)
      extras+=("preprocessing")
      shift
      ;;
    --with-paddleocr)
      extras+=("paddleocr")
      shift
      ;;
    --with-layoutlmv3)
      extras+=("layoutlmv3")
      shift
      ;;
    --dev)
      extras+=("dev")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
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

plan_extras=()
plan_sources=()
plan_models=()
plan_backends=()
if [[ "$mode" != "minimal" ]]; then
  selector=(--all-models)
  if [[ "$mode" == "pipeline" ]]; then
    selector=(--pipeline "${pipeline[@]}")
  fi
  while IFS='=' read -r key value; do
    IFS=',' read -r -a values <<< "$value"
    case "$key" in
      extras) plan_extras=("${values[@]}") ;;
      sources) plan_sources=("${values[@]}") ;;
      models) plan_models=("${values[@]}") ;;
      backends) plan_backends=("${values[@]}") ;;
    esac
  done < <(
    PYTHONPATH="$project_root/src" python -m invoice_ocr.setup_selection \
      "${selector[@]}" --format lines
  )
  extras+=("${plan_extras[@]}")
fi

unique_extras=()
for candidate in "${extras[@]}"; do
  duplicate=false
  for existing in "${unique_extras[@]}"; do
    if [[ "$candidate" == "$existing" ]]; then
      duplicate=true
      break
    fi
  done
  if [[ "$duplicate" == false ]]; then
    unique_extras+=("$candidate")
  fi
done

specifier="$project_root"
if [[ ${#unique_extras[@]} -gt 0 ]]; then
  joined="$(IFS=,; echo "${unique_extras[*]}")"
  specifier="$project_root[$joined]"
fi
python -m pip install -e "$specifier"

for source in "${plan_sources[@]}"; do
  python "$project_root/scripts/fetch_model_sources.py" --source "$source"
done

for model in "${plan_models[@]}"; do
  python "$project_root/scripts/download_models.py" \
    --model "$model" \
    --model-root "$project_root/models"
done

python - <<'PY'
import importlib.util

torch_spec = importlib.util.find_spec("torch")
paddle_spec = importlib.util.find_spec("paddle")
if torch_spec:
    import torch
    print(f"PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}")
else:
    print("PyTorch is not installed; PyTorch backends will fail readiness checks.")
if paddle_spec:
    import paddle
    print(f"PaddlePaddle {paddle.__version__}; compiled_with_cuda={paddle.is_compiled_with_cuda()}")
else:
    print(
        "PaddlePaddle is not installed. Install the official CPU/CUDA-compatible build "
        "before verifying PaddleOCR or VI-LayoutXLM."
    )
PY

if [[ ${#plan_backends[@]} -gt 0 ]]; then
  verify_args=()
  for backend in "${plan_backends[@]}"; do
    verify_args+=(--backend "$backend")
  done
  python -m invoice_ocr.cli verify-models \
    "${verify_args[@]}" \
    --model-root "$project_root/models" \
    --external-root "$project_root/external"
fi

echo "Setup completed without changing the active environment."
echo "Check official PaddlePaddle/PyTorch CUDA matrices before installing GPU runtimes."

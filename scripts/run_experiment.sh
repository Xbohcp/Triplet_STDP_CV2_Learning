#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

EPOCHS=20
BATCH_SIZE=128
LEARNING_RATE=0.01
HIDDEN_LAYERS=3
HIDDEN_DIM=500
TRAIN_RULE="biological"
SCALE_MODE="tanh"
UPDATE_FUNCTION="epsilon_times_activation"
DEVICE=""
LIMIT_SAMPLES=""
DATA_DIR="data"
OUTPUT_CSV="outputs/angle_metrics.csv"
OUTPUT_PLOT="outputs/angle_trend.png"
RUN_TESTS=1
SETUP_ONLY=0
OPEN_OUTPUT=0
UPDATE_BIAS=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_experiment.sh [options] [-- extra train_mnist.py args]

Default:
  Run the full MNIST experiment with a 784 -> 500 -> 500 -> 500 -> 10 network.

Common options:
  --smoke                 Fast sanity check: 1 epoch, 64 samples, batch 32.
  --quick                 Medium run: 2 epochs, 1024 samples, batch 128.
  --setup-only            Create .venv and install requirements, then exit.
  --no-tests              Skip pytest before training.
  --open                  Open the generated plot after training on macOS.

Experiment options:
  --epochs N              Default: 20
  --batch-size N          Default: 128
  --learning-rate X       Default: 0.01
  --hidden-layers N       Default: 3
  --hidden-dim N          Default: 500
  --limit-samples N       Use only the first N MNIST samples.
  --train-rule RULE       biological, bp, or none. Default: biological
  --scale-mode MODE       tanh or clamp. Default: tanh
  --update-function NAME  epsilon_times_activation, epsilon, or activity_gated.
  --update-bias           Also update bias terms.
  --device DEVICE         cpu, cuda, or mps. If omitted, Python script chooses.
  --data-dir PATH         Default: data
  --output-csv PATH       Default: outputs/angle_metrics.csv
  --output-plot PATH      Default: outputs/angle_trend.png

Examples:
  ./scripts/run_experiment.sh --smoke
  ./scripts/run_experiment.sh --epochs 20 --open
  ./scripts/run_experiment.sh --train-rule bp --output-plot outputs/bp_angle.png
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

need_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "${value}" ]] || fail "${option} requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --smoke)
      EPOCHS=1
      BATCH_SIZE=32
      LIMIT_SAMPLES=64
      OUTPUT_CSV="outputs/smoke_metrics.csv"
      OUTPUT_PLOT="outputs/smoke_angle_trend.png"
      ;;
    --quick)
      EPOCHS=2
      BATCH_SIZE=128
      LIMIT_SAMPLES=1024
      OUTPUT_CSV="outputs/quick_metrics.csv"
      OUTPUT_PLOT="outputs/quick_angle_trend.png"
      ;;
    --setup-only)
      SETUP_ONLY=1
      ;;
    --no-tests)
      RUN_TESTS=0
      ;;
    --open)
      OPEN_OUTPUT=1
      ;;
    --update-bias)
      UPDATE_BIAS=1
      ;;
    --epochs)
      need_value "$1" "${2:-}"
      EPOCHS="$2"
      shift
      ;;
    --batch-size)
      need_value "$1" "${2:-}"
      BATCH_SIZE="$2"
      shift
      ;;
    --learning-rate)
      need_value "$1" "${2:-}"
      LEARNING_RATE="$2"
      shift
      ;;
    --hidden-layers)
      need_value "$1" "${2:-}"
      HIDDEN_LAYERS="$2"
      shift
      ;;
    --hidden-dim)
      need_value "$1" "${2:-}"
      HIDDEN_DIM="$2"
      shift
      ;;
    --limit-samples)
      need_value "$1" "${2:-}"
      LIMIT_SAMPLES="$2"
      shift
      ;;
    --train-rule)
      need_value "$1" "${2:-}"
      TRAIN_RULE="$2"
      shift
      ;;
    --scale-mode)
      need_value "$1" "${2:-}"
      SCALE_MODE="$2"
      shift
      ;;
    --update-function)
      need_value "$1" "${2:-}"
      UPDATE_FUNCTION="$2"
      shift
      ;;
    --device)
      need_value "$1" "${2:-}"
      DEVICE="$2"
      shift
      ;;
    --data-dir)
      need_value "$1" "${2:-}"
      DATA_DIR="$2"
      shift
      ;;
    --output-csv)
      need_value "$1" "${2:-}"
      OUTPUT_CSV="$2"
      shift
      ;;
    --output-plot)
      need_value "$1" "${2:-}"
      OUTPUT_PLOT="$2"
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
  shift
done

cd "${PROJECT_DIR}"

log "Project directory: ${PROJECT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  log "Creating virtual environment at ${VENV_DIR}"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "${VENV_DIR}"
  else
    python3 -m venv "${VENV_DIR}"
  fi
fi

log "Installing/updating dependencies"
if command -v uv >/dev/null 2>&1; then
  uv pip install -r requirements.txt
else
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

log "Dependency versions"
"${PYTHON_BIN}" - <<'PY'
import numpy
import torch
import torchvision

print(f"numpy={numpy.__version__}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
PY

if [[ "${SETUP_ONLY}" -eq 1 ]]; then
  log "Setup complete"
  exit 0
fi

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  log "Running tests"
  "${PYTHON_BIN}" -m pytest tests/test_learning_rules.py -q
fi

TRAIN_CMD=(
  "${PYTHON_BIN}" -m triplet_stdp_cv2_learning.train_mnist
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --hidden-layers "${HIDDEN_LAYERS}"
  --hidden-dim "${HIDDEN_DIM}"
  --train-rule "${TRAIN_RULE}"
  --scale-mode "${SCALE_MODE}"
  --update-function "${UPDATE_FUNCTION}"
  --data-dir "${DATA_DIR}"
  --output-csv "${OUTPUT_CSV}"
  --output-plot "${OUTPUT_PLOT}"
)

if [[ -n "${LIMIT_SAMPLES}" ]]; then
  TRAIN_CMD+=(--limit-samples "${LIMIT_SAMPLES}")
fi

if [[ -n "${DEVICE}" ]]; then
  TRAIN_CMD+=(--device "${DEVICE}")
fi

if [[ "${UPDATE_BIAS}" -eq 1 ]]; then
  TRAIN_CMD+=(--update-bias)
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

log "Running experiment"
printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

log "Experiment outputs"
printf 'CSV : %s\n' "${PROJECT_DIR}/${OUTPUT_CSV}"
printf 'Plot: %s\n' "${PROJECT_DIR}/${OUTPUT_PLOT}"

if [[ "${OPEN_OUTPUT}" -eq 1 ]]; then
  if command -v open >/dev/null 2>&1; then
    open "${OUTPUT_PLOT}"
  else
    log "The --open option is only supported when the open command exists"
  fi
fi

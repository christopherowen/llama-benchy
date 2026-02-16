#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required but not found in PATH."
  echo "Install: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

# Create local venv if missing, then activate it.
if [[ ! -d ".venv" ]]; then
  uv venv .venv
fi
source ".venv/bin/activate"

echo "==> Updating llama-benchy editable install with intelligence extras"
uv pip install -e ".[intelligence]"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-gpt-oss-120b}"
OUTPUT_DIR="${OUTPUT_DIR:-./bench_runs}"
FORMAT="${FORMAT:-json}"
INTELLIGENCE_PLUGINS="${INTELLIGENCE_PLUGINS:-all}"
ALLOW_CODE_EXEC="${ALLOW_CODE_EXEC:-1}"
TOKENIZER="${TOKENIZER:-}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-}"
MAX_CONCURRENT="${MAX_CONCURRENT:-}"

read -r -a PLUGIN_ARGS <<< "${INTELLIGENCE_PLUGINS}"

mkdir -p "${OUTPUT_DIR}"
if [[ -n "${DATASET_CACHE_DIR}" ]]; then
  mkdir -p "${DATASET_CACHE_DIR}"
fi

CMD=(
  llama-benchy
  --base-url "${BASE_URL}"
  --model "${MODEL}"
  --enable-intelligence
  --intelligence-plugins "${PLUGIN_ARGS[@]}"
  --output-dir "${OUTPUT_DIR}"
  --format "${FORMAT}"
)

if [[ "${ALLOW_CODE_EXEC}" == "1" ]]; then
  CMD+=(--allow-code-exec)
fi

if [[ -n "${TOKENIZER}" ]]; then
  CMD+=(--tokenizer "${TOKENIZER}")
fi

if [[ -n "${DATASET_CACHE_DIR}" ]]; then
  CMD+=(--dataset-cache-dir "${DATASET_CACHE_DIR}")
fi

if [[ -n "${MAX_CONCURRENT}" ]]; then
  CMD+=(--max-concurrent "${MAX_CONCURRENT}")
fi

# Allow callers to append/override any flags.
CMD+=("$@")

echo "==> Running benchmark command"
printf ' %q' "${CMD[@]}"
echo
exec "${CMD[@]}"

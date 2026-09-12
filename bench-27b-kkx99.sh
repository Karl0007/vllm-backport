#!/usr/bin/env bash
# Driver for the OFFICIAL `vllm bench serve` (from the backport image) against the
# local 27B server. Not a reimplementation: it only fixes the knobs that make runs
# comparable — random dataset (no prefix-cache contamination), greedy sampling,
# concurrency 1 for latency, and saved JSON per label.
#
# usage: bench-27b-kkx99.sh <label> [scenario...]     scenarios: decode prefill batch
set -euo pipefail

LABEL="${1:?label required}"; shift || true
if [ "$#" -gt 0 ]; then SCENARIOS=("$@"); else SCENARIOS=(decode prefill batch); fi
IMAGE="${IMAGE:-vllm/vllm-backport:cmp170hx}"
MODEL_DIR="${MODEL_DIR:-/home/kk/models/llm/qwen3.8-27b-awq-int4}"
SERVED="${SERVED:-qwen3.8-27b-awq-int4}"
PORT="${PORT:-18000}"
OUT_DIR="${OUT_DIR:-/tmp/bench-out}"
mkdir -p "$OUT_DIR"

run() { # <name> <input-len> <output-len> <num-prompts> <max-concurrency>
  echo "--- $LABEL / $1: in=$2 out=$3 n=$4 conc=$5"
  # --gpus is required even for the client: vllm's import chain pulls in triton
  # (vllm/v1/attention/ops/fp8_sm80.py does `tl.constexpr(...)` at import time),
  # and without a visible device `triton.language` is None -> TypeError at import.
  sudo docker run --rm --network host --gpus "device=${BENCH_GPU:-1}" \
    -v "${MODELS_DIR:-/home/kk/models/llm}":/models:ro -v "$OUT_DIR":/bench-out --entrypoint vllm "$IMAGE" bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --base-url "http://127.0.0.1:$PORT" \
    --model "/models/$(basename "$MODEL_DIR")" --served-model-name "$SERVED" \
    --dataset-name random --random-input-len "$2" --random-output-len "$3" \
    --random-range-ratio 0.0 --random-prefix-len 0 \
    --num-prompts "$4" --max-concurrency "$5" --request-rate inf \
    --temperature 0 --top-p 1.0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir /bench-out --result-filename \
    "${LABEL}--${1}.json" --label "${LABEL}-${1}" --seed 1234 --disable-tqdm 2>&1 |
    grep -aE "Successful requests|Benchmark duration|Output token throughput|Request throughput|Total token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL|Acceptance rate|Acceptance length|Per-position|Traceback|Error|error:" || true
}

for s in "${SCENARIOS[@]}"; do
  case "$s" in
    decode)  run decode  128  512 12 1 ;;   # decode-bound, single stream
    # Scaling ladder: if aggregate throughput grows ~linearly with concurrency,
    # the per-step CPU (X99 @2.3 GHz) is the single-stream ceiling, not the GPU.
    c2)      run c2      128  512 16 2 ;;
    c4)      run c4      128  512 24 4 ;;
    c8)      run c8      128  512 32 8 ;;
    prefill) run prefill 8192  32  8 1 ;;   # prefill-bound, cache-miss by construction
    batch)   run batch   1024 256 24 4 ;;   # 4-way concurrency: does speculation still pay?
    *) echo "unknown scenario $s" >&2; exit 1 ;;
  esac
done

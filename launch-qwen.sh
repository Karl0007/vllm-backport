#!/usr/bin/env bash
# Qwen3.8-Flash-Next-FP8 on the unified vllm-backport base (sm80: Marlin MoE,
# BF16 KV, explicit CUDA_VISIBLE_DEVICES). Mirrors /opt/qwen38-deploy/
# deploy_qwen38.sh MODE=pp|tp, but on the fork (PP unlock = upstream PR#47).
#
# MODE=pp (target; needs >=90 GiB host RAM):
#   TP1 + PP4 + VLLM_PLE_CPU_OFFLOAD=1. The 47.7 GiB FP8 n-gram table is
#   pinned in host RAM by the PleOffloadWorker on stage 0; per-card ~31 GiB
#   weights + ~9 GiB overhead -> ~17 GiB KV/card, 262K ctx, no per-layer
#   allreduce. MTP is ON (NUM_SPEC=3): PR#47 relays the target hidden states
#   to the draft head on the last PP rank (validated TP1xPP4 on this same
#   4x CMP 170HX 64GB). If the vllm#53896 CUDA IMA (query_len>1 intermediate
#   ranks, seen on the old qwen38-pp stack) still reproduces, set NUM_SPEC=0.
#
# MODE=tp (fallback): TP4 + EP + MTP3; ngram table in VRAM when RAM < 90 GiB
#   (ctx 65K), offloaded to host RAM (ctx 262K) when RAM >= 90 GiB.
set -euo pipefail

MODEL_HOST_PATH="${MODEL_HOST_PATH:-/srv/models/Qwen/Qwen3.8-Flash-Next-FP8}"
MODEL_CONTAINER_PATH="/models/qwen38"
PORT="${PORT:-8010}"
API_KEY="${API_KEY:-}"
IMAGE="${IMAGE:-vllm/vllm-backport:cmp170hx}"
GMU="${GMU:-0.90}"
NUM_SPEC="${NUM_SPEC:-3}"
MODE="${MODE:-pp}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_SEQS="${MAX_SEQS:-8}"
PP_PARTITION="${PP_PARTITION:-12,12,12,12}"
NAME="${NAME:-qwen38-cmp170hx}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next-FP8}"
CACHE_DIR="$(dirname "$0")/vllm_cache_qwen"

MEM_GB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
PLE_ENV=(); PARALLEL_ARGS=(); MAXLEN_DEFAULT=65536
if [ "$MODE" = "pp" ]; then
  [ "$MEM_GB" -ge 90 ] || { echo "PP mode needs >=90 GiB RAM (PLE offload ngram table 47.7 GiB); have ${MEM_GB}. Use MODE=tp." >&2; exit 1; }
  PLE_ENV=(-e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800)
  MAXLEN_DEFAULT=262144
  PARALLEL_ARGS=(--tensor-parallel-size 1 --pipeline-parallel-size 4 --moe-backend marlin --async-scheduling)
elif [ "$MODE" = "tp" ]; then
  PARALLEL_ARGS=(--tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin)
  if [ "$MEM_GB" -ge 90 ]; then
    PLE_ENV=(-e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800)
    MAXLEN_DEFAULT=262144
  fi
else
  echo "unknown MODE=${MODE} (must be pp|tp)" >&2
  exit 1
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-$MAXLEN_DEFAULT}"

SPEC_ARGS=()
if [ "$NUM_SPEC" -gt 0 ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC}")
fi

mkdir -p "${CACHE_DIR}/triton" "${CACHE_DIR}/tilelang"

serve() {
sudo docker run --rm \
  --name "$NAME" \
  --gpus all --ipc=host --privileged \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_PP_LAYER_PARTITION="$PP_PARTITION" \
  -e PYTHONUNBUFFERED=1 -e VLLM_LOGGING_LEVEL=INFO \
  "${PLE_ENV[@]}" \
  -p "$PORT:8000" \
  -v "$MODEL_HOST_PATH":"$MODEL_CONTAINER_PATH":ro \
  -v "$CACHE_DIR":/root/.cache/vllm \
  -v "$CACHE_DIR/triton":/root/.cache/triton \
  -v "$CACHE_DIR/tilelang":/root/.cache/tilelang \
  "$IMAGE" \
  "$MODEL_CONTAINER_PATH" \
  --served-model-name "$SERVED_NAME" \
  ${API_KEY:+--api-key} ${API_KEY:+"$API_KEY"} \
  "${PARALLEL_ARGS[@]}" \
  --gpu-memory-utilization "$GMU" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  "${SPEC_ARGS[@]}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  "$@"
}

serve "$@" &
CPID=$!
# Orchestrator stops the worker's process group; make sure the (detached)
# docker CLI does not keep the --rm container running after we exit.
trap 'sudo docker stop "$NAME" >/dev/null 2>&1 || true' TERM INT
wait "$CPID"
RC=$?
sudo docker stop "$NAME" >/dev/null 2>&1 || true
exit $RC

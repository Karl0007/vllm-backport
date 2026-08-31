#!/usr/bin/env bash
# Qwen3.8-Flash-Next-FP8 TP4+EP+MTP3 smoke on the unified vllm-backport base.
# Mirrors /opt/qwen38-deploy/deploy_qwen38.sh MODE=tp (the only mode runnable
# on 62 GiB RAM; PLE CPU offload needs >=90). sm80: Marlin MoE, BF16 KV,
# explicit CUDA_VISIBLE_DEVICES. Adds the fork README's rope yarn hf-overrides
# (long-ctx scaling) and FULL_AND_PIECEWISE cudagraphs.
set -euo pipefail

MODEL_HOST_PATH="${MODEL_HOST_PATH:-/srv/models/Qwen/Qwen3.8-Flash-Next-FP8}"
MODEL_CONTAINER_PATH="/models/qwen38"
PORT="${PORT:-8010}"
API_KEY="${API_KEY:-}"
IMAGE="${IMAGE:-vllm/vllm-backport:cmp170hx}"
GMU="${GMU:-0.90}"
NUM_SPEC="${NUM_SPEC:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
CACHE_DIR="$(dirname "$0")/vllm_cache_qwen"

mkdir -p "${CACHE_DIR}/triton" "${CACHE_DIR}/tilelang"

serve() {
sudo docker run --rm \
  --name qwen38-cmp170hx \
  --gpus all --ipc=host --privileged \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTHONUNBUFFERED=1 -e VLLM_LOGGING_LEVEL=INFO \
  -p "$PORT:8000" \
  -v "$MODEL_HOST_PATH":"$MODEL_CONTAINER_PATH":ro \
  -v "$CACHE_DIR":/root/.cache/vllm \
  -v "$CACHE_DIR/triton":/root/.cache/triton \
  -v "$CACHE_DIR/tilelang":/root/.cache/tilelang \
  "$IMAGE" \
  "$MODEL_CONTAINER_PATH" \
  --served-model-name Qwen3.8-Flash-Next-FP8 \
  ${API_KEY:+--api-key} ${API_KEY:+"$API_KEY"} \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --moe-backend marlin \
  --gpu-memory-utilization "$GMU" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC}" \
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
trap 'sudo docker stop qwen38-cmp170hx >/dev/null 2>&1 || true' TERM INT
wait "$CPID"
RC=$?
sudo docker stop qwen38-cmp170hx >/dev/null 2>&1 || true
exit $RC

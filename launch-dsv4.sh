#!/usr/bin/env bash
# DSV4-Flash PP4+DSpark smoke on the unified vllm-backport base.
# Flags mirror /opt/vllm170hx/launch/start-cmp170hx.sh (production, 98 tok/s
# decode on the old f8ea5bb line), including the 1M context (ROW_CHUNK=64 is
# the prerequisite for cumulative chats >700k, see handoff doc).
set -euo pipefail

MODEL_HOST_PATH="${MODEL_HOST_PATH:-/srv/models/DeepSeek-V4-Flash-0731}"
MODEL_CONTAINER_PATH="/models/DeepSeek-V4-Flash-0731"
PORT="${PORT:-8011}"
API_KEY="${API_KEY:-}"
# B5(fail-open 修复)：API key 未设置时拒绝启动而不是起一个 0.0.0.0 发布、完全无鉴权的 vLLM。
if [ -z "$API_KEY" ]; then
  echo "ERROR: API_KEY 未设置（orchestrator profile 的 env 应该传它）——拒绝启动无鉴权 worker" >&2
  exit 1
fi
IMAGE="${IMAGE:-vllm/vllm-backport:cmp170hx}"
PP="${PP:-4}"
PP_PARTITION="${PP_PARTITION:-12,12,12,7}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"
GMU="${GMU:-0.85}"
MAX_SEQS="${MAX_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
CACHE_DIR="$(dirname "$0")/vllm_cache_dsv4"

mkdir -p "${CACHE_DIR}/triton" "${CACHE_DIR}/tilelang"

serve() {
sudo docker run --rm \
  --name dsv4-cmp170hx \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --privileged --ipc=host \
  -p 127.0.0.1:${PORT}:8000 \
  -v "${MODEL_HOST_PATH}:${MODEL_CONTAINER_PATH}:ro" \
  -v "${CACHE_DIR}:/root/.cache/vllm" \
  -v "${CACHE_DIR}/triton:/root/.cache/triton" \
  -v "${CACHE_DIR}/tilelang:/root/.cache/tilelang" \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PP_LAYER_PARTITION=${PP_PARTITION} \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTHONUNBUFFERED=1 \
  -e VLLM_LOGGING_LEVEL=INFO \
  ${IMAGE} \
  "${MODEL_CONTAINER_PATH}" \
  --served-model-name deepseek-v4-flash-0731 \
  ${API_KEY:+--api-key} ${API_KEY:+"$API_KEY"} \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size ${PP} \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens 2048 \
  --max-num-seqs ${MAX_SEQS} \
  --gpu-memory-utilization ${GMU} \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\"}" \
  "$@"
}

serve "$@" &
CPID=$!
# Orchestrator stops the worker's process group; make sure the (detached)
# docker CLI does not keep the --rm container running after we exit.
trap 'sudo docker stop dsv4-cmp170hx >/dev/null 2>&1 || true' TERM INT
wait "$CPID"
RC=$?
sudo docker stop dsv4-cmp170hx >/dev/null 2>&1 || true
exit $RC

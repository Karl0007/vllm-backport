#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 PP4+MTP smoke on the unified vllm-backport base.
# Flags mirror the live reference container (glm53-flash-8008, partition
# 14,11,11,9) so the only variable is the runtime image (fork cmp170hx =
# v0.11.1-sm80 wheel + PR#44 + ported GLM PP/MTP/Qwen-PP Python patches).
set -euo pipefail

MODEL_HOST_PATH="${MODEL_HOST_PATH:-/srv/models/RedHatAI/GLM-5.3-Flash-NVFP4}"
MODEL_CONTAINER_PATH="/models/GLM-5.3-Flash-NVFP4"
PORT="${PORT:-8009}"
API_KEY="${API_KEY:-}"
# B5(fail-open 修复)：API key 未设置时拒绝启动而不是起一个 0.0.0.0 发布、完全无鉴权的 vLLM。
if [ -z "$API_KEY" ]; then
  echo "ERROR: API_KEY 未设置（orchestrator profile 的 env 应该传它）——拒绝启动无鉴权 worker" >&2
  exit 1
fi
IMAGE="${IMAGE:-vllm/vllm-backport:cmp170hx}"
TP="${TP:-1}"
PP="${PP:-4}"
PP_PARTITION="${PP_PARTITION:-14,11,11,9}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"
GMU="${GMU:-0.95}"
MAX_SEQS="${MAX_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
# NVFP4 (RedHatAI) needs moe-backend marlin (unquantized-MoE path); AWQ W4A16
# checkpoints carry quantized MoE and must NOT pass it -> override with MOE_BACKEND=
MOE_BACKEND="${MOE_BACKEND-marlin}"
CACHE_DIR="$(dirname "$0")/vllm_cache"
NAME="${NAME:-glm53-cmp170hx}"
SERVED_NAME="${SERVED_NAME:-glm-5.3-flash-nvfp4}"

SPEC_ARGS=()
if [ "${NUM_SPEC_TOKENS}" -gt 0 ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}")
fi

mkdir -p "${CACHE_DIR}/triton" "${CACHE_DIR}/tilelang"

serve() {
sudo docker run --rm \
  --name ${NAME} \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --privileged --ipc=host \
  -p 127.0.0.1:${PORT}:8000 \
  -v "${MODEL_HOST_PATH}:${MODEL_CONTAINER_PATH}:ro" \
  -v "${CACHE_DIR}:/root/.cache/vllm" \
  -v "${CACHE_DIR}/triton:/root/.cache/triton" \
  -v "${CACHE_DIR}/tilelang:/root/.cache/tilelang" \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE \
  -e VLLM_PP_LAYER_PARTITION=${PP_PARTITION} \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTHONUNBUFFERED=1 \
  -e VLLM_LOGGING_LEVEL=INFO \
  ${IMAGE} \
  "${MODEL_CONTAINER_PATH}" \
  --served-model-name ${SERVED_NAME} \
  ${API_KEY:+--api-key} ${API_KEY:+"$API_KEY"} \
  --tensor-parallel-size ${TP} \
  --pipeline-parallel-size ${PP} \
  --enable-expert-parallel \
  ${MOE_BACKEND:+--moe-backend ${MOE_BACKEND}} \
  --gpu-memory-utilization ${GMU} \
  --max-num-seqs ${MAX_SEQS} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  "${SPEC_ARGS[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --attention-config '{"sparse_mla_force_mqa": true}' \
  "$@"
}

serve "$@" &
CPID=$!
# Orchestrator stops the worker's process group; make sure the (detached)
# docker CLI does not keep the --rm container running after we exit.
trap 'sudo docker stop ${NAME} >/dev/null 2>&1 || true' TERM INT
wait "$CPID"
RC=$?
sudo docker stop ${NAME} >/dev/null 2>&1 || true
exit $RC

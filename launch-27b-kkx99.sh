#!/usr/bin/env bash
# Qwen3.8-27B on the sm_80 backport stack — ONE CMP 170HX (64 GiB), TP1.
#
# Config baseline is "what an A100-class card should run", not what the sm75
# fork needed:
#   * bf16 compute + BF16 KV. FP8 KV-cache is only validated upstream on
#     Hopper(FA3)/Blackwell(FlashInfer) and vllm#43914 gates it at sm89+, so
#     fp8_e4m3 KV (inherited from the 22 GiB era) is not a valid option here.
#   * AWQ/GPTQ 4-bit is the Ampere weight-only track; FP8 checkpoints degrade
#     to Marlin W8A16 emulation on sm80.
#   * `--mamba-cache-mode align` is required by the hybrid GDN arch
#     (qwen3_5.py rejects `all`).
#   * No CPU-offload features: this host has 30 GiB RAM (PP/PLE need >=90 GiB).
#
# Everything perf-relevant is env-overridable so the same script drives the
# whole benchmark matrix (AWQ vs W8A8, none/MTP/DFlash2, batch sizes).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-vllm/vllm-backport:cmp170hx}"
MODEL="${MODEL:-/home/kk/models/llm/qwen3.8-27b-awq-int4}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b-awq-int4}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
PORT="${PORT:-18000}"
NAME="${NAME:-orchestrator-vllm-27b}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GMU="${GMU:-0.92}"
MAX_SEQS="${MAX_SEQS:-8}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
DTYPE="${DTYPE:-bfloat16}"
KV_DTYPE="${KV_DTYPE:-auto}"
SSM_DTYPE="${SSM_DTYPE:-}"          # empty = checkpoint default (fp32); float16 = 22 GiB-era trick
GDN_BACKEND="${GDN_BACKEND:-}"      # empty = auto; flashinfer | triton | cutedsl
MOE_BACKEND="${MOE_BACKEND:-}"
if [ -z "${COMPILE_CFG:-}" ]; then COMPILE_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'; fi
SPEC="${SPEC:-none}"                # none | mtp | dflash
NUM_SPEC="${NUM_SPEC:-7}"           # DFlash2 block_size 8 => 7 draft tokens
DRAFT="${DRAFT:-/home/kk/models/llm/qwen3.8-27b-dflash2}"
API_KEY="${API_KEY:-}"

CACHE_DIR="${CACHE_DIR:-$(dirname "$0")/kkx99-cache}"

# Orchestrator restarts call this repeatedly; a --rm container from a previous
# lifecycle may still be in "removing" state. Stop+wait so --name does not collide.
if sudo docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  sudo docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  sleep 2
fi
TEMPLATES="${TEMPLATES:-/home/kk/services/model-runtimes/vllm/templates}"

EXTRA=()
# W4A8: keep int4 weights, quantize activations per-token to int8 and run the
# Marlin GEMMs on Ampere's int8 tensor cores. Upstream env (envs.py:199), and the
# recipe's measured prefill win for it is +19..30% (decode at batch 1 is
# memory-bound, so it does nothing there). INT8_ACT_LAYERS maps to syv's
# VLLM_MARLIN_INT8_INCLUDE_RE, which this checkout does not have — empty means
# every Marlin layer (their "all" tier). These must reach the CONTAINER, not the
# host shell, since vLLM runs inside it.
DOCKER_ENV=()
if [ -n "${EXTRA_ENV:-}" ]; then DOCKER_ENV+=($EXTRA_ENV); fi
# Split-KV spec-decode verify attention (vllm/v1/attention/ops/spec_decode_attn.py,
# hooked in the FlashAttention backend). vLLM's FA2 and Triton paths both refuse to
# split the KV sequence when a request has more than one query token, so a verify
# step reads the whole KV with num_kv_heads thread blocks: 107 ms/step at 110k versus
# 41 ms with this kernel, i.e. 19.9 -> 54.9 tok/s (BENCH-27b-kkx99.md, 2026-09-12).
if [ "${SPEC_ATTN:-0}" = "1" ]; then DOCKER_ENV+=(-e VLLM_SPEC_DECODE_ATTN=1); fi
if [ -n "${SPEC_ATTN_QMAX:-}" ]; then DOCKER_ENV+=(-e "VLLM_SPEC_DECODE_ATTN_QMAX=$SPEC_ATTN_QMAX"); fi
if [ -n "${INT8_ACT:-}" ]; then DOCKER_ENV+=(-e "VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT"); fi
if [ -n "${INT8_ACT_LAYERS:-}" ]; then DOCKER_ENV+=(-e "VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_ACT_LAYERS"); fi
# Overlap CPU scheduling with GPU execution. On a 2.3 GHz X99 the per-step
# scheduler/sample work is a candidate single-stream ceiling, so this is a
# measured option rather than a default (cmp170hx's launcher enables it).
if [ "${ASYNC_SCHED:-0}" = "1" ]; then EXTRA+=(--async-scheduling); fi
if [ -n "$SSM_DTYPE" ]; then EXTRA+=(--mamba-ssm-cache-dtype "$SSM_DTYPE"); fi
if [ -n "$GDN_BACKEND" ]; then EXTRA+=(--gdn-prefill-backend "$GDN_BACKEND"); fi
if [ -n "$MOE_BACKEND" ]; then EXTRA+=(--moe-backend "$MOE_BACKEND"); fi

SPEC_ARGS=()
# Run this checkout's python sources over the image's installed copy. The image
# (vllm/vllm-backport:cmp170hx-v013-nopatch) was built from this checkout at
# v0.13.0, so a single-file bind mount is byte-compatible apart from the patch
# itself. Mount only files that actually differ from the image — a mount whose
# copy is identical is dead weight that hides drift.
#   qwen3_5*.py       PATCH_DFLASH_QUANT_DRAFTER / embed-quant hunks
#   qwen3_dflash.py   vllm#51620 (DFlash with a weight-quantized drafter)
#   flash_attn.py     split-KV spec-decode hook (SPEC_ATTN=1)
#   spec_decode_attn.py  the kernel itself
# Used today for PATCH_DFLASH_QUANT_DRAFTER=1 = vllm#51620 (DFlash with a
# weight-quantized drafter; without it the W4A16 draft dies at load with
# "'QKVParallelLinear' object has no attribute 'weight'").
PATCH_ARGS=()
# The two qwen3_5 files carry the embed-quant hunk (vLLM has a
# dequant-on-gather kernel for int-quantized embedding tables but qwen3_5 never
# passed quant_config to VocabParallelEmbedding; without it an eq8emb-style
# checkpoint dies at load with "no parameter named 'embed_tokens.weight_packed'").
# Mount them unconditionally — they are byte-identical to the image for a
# non-quantized-embed checkpoint, the hunk only activates when the checkpoint
# asks for a packed embedding table.
for _m in qwen3_5.py qwen3_5_mtp.py; do
  PATCH_ARGS+=(-v "$HERE/vllm/model_executor/models/$_m:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/$_m:ro")
done
# Split-KV spec-decode attention + the backends it plugs into: same convention as
# the model files above (this checkout's copy wins over the image's), so a kernel
# tweak is a container restart rather than an image rebuild.
for _f in "v1/attention/ops/spec_decode_attn.py" "v1/attention/backends/flash_attn.py" "v1/core/single_type_kv_cache_manager.py" "v1/worker/gpu/model_states/mamba_hybrid.py" "v1/worker/mamba_utils.py" "model_executor/models/qwen3_next.py" "v1/sample/rejection_sampler.py" "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"; do
  PATCH_ARGS+=(-v "$HERE/vllm/$_f:/usr/local/lib/python3.12/dist-packages/vllm/$_f:ro")
done
if [ "${PATCH_DFLASH_QUANT_DRAFTER:-0}" = "1" ]; then
  PATCH_ARGS+=(-v "$HERE/vllm/model_executor/models/qwen3_dflash.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py:ro")
fi
case "$SPEC" in
  none) ;;
  # draft_sample_method=probabilistic lifts draft acceptance at temperature > 0
  # (the recipe's note); greedy stays correct for temperature 0.
  mtp) SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC${DRAFT_SAMPLE:+,\"draft_sample_method\":\"$DRAFT_SAMPLE\"}}") ;;
  dflash) SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/models/$(basename "$DRAFT")\",\"num_speculative_tokens\":$NUM_SPEC${DRAFT_SAMPLE:+,\"draft_sample_method\":\"$DRAFT_SAMPLE\"}}") ;;
  *) echo "SPEC must be none|mtp|dflash, got $SPEC" >&2; exit 1 ;;
esac

mkdir -p "$CACHE_DIR/vllm" "$CACHE_DIR/triton" "$CACHE_DIR/inductor"

sudo docker run --rm \
  --name "$NAME" \
  --gpus "device=$GPU_ID" \
  --ipc=host \
  --shm-size=4g \
  -e PYTHONUNBUFFERED=1 \
  "${DOCKER_ENV[@]}" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p "127.0.0.1:$PORT:8000" \
  -v "$(dirname "$MODEL")":/models:ro \
  "${PATCH_ARGS[@]}" \
  -v "$TEMPLATES":/templates:ro \
  -v "$CACHE_DIR/vllm":/root/.cache/vllm \
  ${VLLM_MAMBA_DIAG_MOUNT:+-v /dev/shm/vllm-diag:/dev/shm/vllm-diag} \
  -v "$CACHE_DIR/triton":/root/.cache/triton \
  -v "$CACHE_DIR/inductor":/root/.cache/torchinductor \
  "$IMAGE" \
  /models/"$(basename "$MODEL")" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port 8000 \
  --dtype "$DTYPE" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --gpu-memory-utilization "$GMU" \
  --kv-cache-dtype "$KV_DTYPE" \
  --mamba-cache-mode align \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":50,"video":0,"audio":0}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --chat-template /templates/qwen-fixed-v22.1.jinja \
  --compilation-config "${COMPILE_CFG}" \
  "${EXTRA[@]}" \
  "${SPEC_ARGS[@]}" \
  ${API_KEY:+--api-key "$API_KEY"} \
  "$@"

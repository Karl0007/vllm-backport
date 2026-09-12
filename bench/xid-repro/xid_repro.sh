#!/usr/bin/env bash
# Standard reproduction harness for the GLM v0.13 PP4/async mixed long-context
# Xid 13. Detection/forensics live in /opt/xid-watch; this script owns the
# *reproduction* half so every fix candidate is judged by the same procedure.
#
# Recipe (from /srv/docs/cmp170hx/glm-v013-kda-oob-crash.md): one streaming
# decode is already running when a 143182-token prefill is submitted; the
# prefill's second-to-last chunk trips the fault within ~3-20 s.
#
# The server must run the PRODUCTION path. Any instrumentation that inserts a
# device synchronization at every step hides the fault (measured: a per-call
# .item() in the indexer turned the crash into a pass), so callers must not add
# host syncs on the hot path.
#
# Usage:
#   xid_repro.sh [rounds]            # default 3 rounds
# Environment:
#   IMAGE=...          (default vllm/vllm-backport:cmp170hx)
#   PORT=...           (default 8899)
#   NAME=...           (default xidrepro)
#   EXTRA_DOCKER_ENV=  e.g. "-e GLM_XID_TRACE=1 -e GLM_XID_SYNC_FILE=/tmp/syncpoint"
#   MODEL_HOST_PATH=...
#   READY_TIMEOUT=...  (default 900 s)
#   ROUND_TIMEOUT=...  (default 300 s)
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROMPT_FILE="${PROMPT_FILE:-${HERE}/prompt.txt}"

IMAGE="${IMAGE:-vllm/vllm-backport:cmp170hx}"
PORT="${PORT:-8899}"
NAME="${NAME:-xidrepro}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/srv/models/wtdcode/GLM-5.3-Flash-AWQ-W4A16}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
ROUND_TIMEOUT="${ROUND_TIMEOUT:-300}"
ROUNDS="${1:-3}"
XID_LOG=/opt/model-runtime/orchestrator/logs/xid-events.jsonl
OUTDIR="${HERE}/runs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUTDIR}"

[ -s "${PROMPT_FILE}" ] || { echo "missing ${PROMPT_FILE}" >&2; exit 2; }

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Build the request payload from the prompt file (single source of truth).
python3 - "$PROMPT_FILE" "${OUTDIR}/request_long.json" <<'PY'
import json, sys
prompt = open(sys.argv[1]).read()
import os
json.dump(
    {
        "model": "glm-5.3-flash-awq",
        "prompt": prompt,
        "max_tokens": int(os.environ.get("MAX_TOKENS", "16")),
        "temperature": 0,
    },
    open(sys.argv[2], "w"),
)
PY

pause_and_clean() {
  if sudo docker inspect "${NAME}" >/dev/null 2>&1; then
    log "stopping container ${NAME}"
    sudo docker stop "${NAME}" >/dev/null 2>&1
    sudo docker rm -f "${NAME}" >/dev/null 2>&1
  fi
}

log "image=${IMAGE} port=${PORT} rounds=${ROUNDS}"
log "artifacts under ${OUTDIR}"
pause_and_clean

# Production-equivalent configuration: the point is to exercise align-mode
# prefix caching + MTP5 + PP4 with async scheduling enabled (default).
sudo docker run -d --rm --name "${NAME}" \
  --gpus all --ipc=host --privileged \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_PP_LAYER_PARTITION=14,12,12,7 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  ${EXTRA_DOCKER_ENV:-} \
  ${EXTRA_DOCKER_ARGS:-} \
  -v "${MODEL_HOST_PATH}:/models/GLM-5.3-Flash-NVFP4:ro" \
  -v /opt/vllm-backport/vllm_cache:/root/.cache/vllm \
  -p "127.0.0.1:${PORT}:8000" \
  "${IMAGE}" \
  /models/GLM-5.3-Flash-NVFP4 \
  --served-model-name glm-5.3-flash-awq \
  --api-key repro-key \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 4 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 8 \
  --max-model-len 524288 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --attention-config '{"sparse_mla_force_mqa": true}' \
  >"${OUTDIR}/docker_run.log" 2>&1

sudo docker logs -f "${NAME}" >"${OUTDIR}/server.log" 2>&1 &
LOG_PID=$!
trap 'kill ${LOG_PID} >/dev/null 2>&1; pause_and_clean' EXIT

log "waiting for /health (timeout ${READY_TIMEOUT}s)"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if [ "$(date +%s)" -gt "${deadline}" ]; then
    log "FAIL: server not ready in ${READY_TIMEOUT}s"
    tail -40 "${OUTDIR}/server.log"
    exit 3
  fi
  if ! sudo docker inspect -f '{{.State.Running}}' "${NAME}" 2>/dev/null | grep -q true; then
    log "FAIL: container exited during startup"
    tail -40 "${OUTDIR}/server.log"
    exit 4
  fi
  sleep 5
done
log "server ready"

xid_count() { [ -f "${XID_LOG}" ] && grep -c 'Xid' "${XID_LOG}" || echo 0; }

verdict=0
for round in $(seq 1 "${ROUNDS}"); do
  before=$(xid_count)
  log "round ${round}: launching streaming decode + 143182-token prefill"
  curl -sS --no-buffer --max-time "${ROUND_TIMEOUT}" \
    -H 'Authorization: Bearer repro-key' -H 'Content-Type: application/json' \
    "http://127.0.0.1:${PORT}/v1/completions" \
    -d '{"model":"glm-5.3-flash-awq","prompt":"Explain in great detail how a transformer works.","max_tokens":3000,"temperature":0.7,"stream":true}' \
    >"${OUTDIR}/round${round}_decode.sse" 2>&1 &
  decode_pid=$!
  sleep 3
  curl -sS --max-time "${ROUND_TIMEOUT}" \
    -H 'Authorization: Bearer repro-key' -H 'Content-Type: application/json' \
    "http://127.0.0.1:${PORT}/v1/completions" \
    --data-binary "@${OUTDIR}/request_long.json" \
    >"${OUTDIR}/round${round}_long.json" 2>&1
  long_rc=$?
  kill "${decode_pid}" >/dev/null 2>&1
  sleep 2
  after=$(xid_count)
  running=$(sudo docker inspect -f '{{.State.Running}}' "${NAME}" 2>/dev/null || echo false)

  prompt_tokens=$(python3 - "${OUTDIR}/round${round}_long.json" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    print(json.load(open(sys.argv[1]))["usage"]["prompt_tokens"])
except Exception:
    print(0)
PY
)
  if [ "${after}" -gt "${before}" ]; then
    log "round ${round}: FAIL — new Xid event (${before} -> ${after}), long_rc=${long_rc}"
    verdict=1
    break
  fi
  if [ "${running}" != "true" ]; then
    log "round ${round}: FAIL — container died (long_rc=${long_rc})"
    verdict=1
    break
  fi
  if [ "${long_rc}" -ne 0 ] || [ "${prompt_tokens}" -ne "${EXPECT_PROMPT_TOKENS:-143182}" ]; then
    log "round ${round}: FAIL — bad response (long_rc=${long_rc}, prompt_tokens=${prompt_tokens})"
    verdict=1
    break
  fi
  if [ -n "${EXPECT_TEXT:-}" ]; then
    if ! grep -q "${EXPECT_TEXT}" "${OUTDIR}/round${round}_long.json"; then
      log "round ${round}: FAIL — expected '${EXPECT_TEXT}' missing from response"
      verdict=1
      break
    fi
    log "round ${round}: needle present"
  fi
  log "round ${round}: pass (prompt_tokens=${prompt_tokens})"
done

if [ "${verdict}" -eq 0 ]; then
  log "RESULT: PASS — ${ROUNDS}/${ROUNDS} rounds clean, no Xid"
else
  log "RESULT: FAIL — see ${OUTDIR}"
fi
exit "${verdict}"

#!/usr/bin/env bash
# Crash-recipe stress against an ALREADY RUNNING server (production path).
#
# xid_repro.sh owns the container lifecycle; this one takes a base URL so the
# same recipe can be pointed at the orchestrator front door (:8091) or the
# public edge. Judge = no new Xid event, container alive, responses sane.
#
# Usage: xid_repro_prod.sh [rounds]
# Env: BASE_URL (default http://127.0.0.1:8091), API_KEY, MAX_TOKENS (300),
#      ROUND_TIMEOUT (600), PROMPT_FILE, EXPECT_TEXT, EXPECT_PROMPT_TOKENS
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8091}"
API_KEY="${API_KEY:?set API_KEY}"
MAX_TOKENS="${MAX_TOKENS:-300}"
ROUND_TIMEOUT="${ROUND_TIMEOUT:-600}"
PROMPT_FILE="${PROMPT_FILE:-${HERE}/prompt.txt}"
EXPECT_PROMPT_TOKENS="${EXPECT_PROMPT_TOKENS:-143182}"
MODEL="${MODEL:-glm-5.3-flash-awq}"
ROUNDS="${1:-3}"
XID_LOG=/opt/model-runtime/orchestrator/logs/xid-events.jsonl
OUTDIR="${HERE}/runs/prod_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUTDIR}"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
xid_count() { [ -f "${XID_LOG}" ] && grep -c Xid "${XID_LOG}" || echo 0; }

python3 - "$PROMPT_FILE" "${OUTDIR}/request_long.json" "$MAX_TOKENS" <<'PY'
import json, sys
json.dump(
    {
        "model": "glm-5.3-flash-awq",
        "prompt": open(sys.argv[1]).read(),
        "max_tokens": int(sys.argv[3]),
        "temperature": 0,
    },
    open(sys.argv[2], "w"),
)
PY

verdict=0
for round in $(seq 1 "${ROUNDS}"); do
  before=$(xid_count)
  log "round ${round}: concurrent stream-decode + $(wc -c <"${PROMPT_FILE}")B prefill"
  curl -sS --no-buffer --max-time "${ROUND_TIMEOUT}" \
    -H "Authorization: Bearer ${API_KEY}" -H 'Content-Type: application/json' \
    "${BASE_URL}/v1/completions" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"Explain in great detail how a transformer works.\",\"max_tokens\":3000,\"temperature\":0.7,\"stream\":true}" \
    >"${OUTDIR}/round${round}_decode.sse" 2>&1 &
  decode_pid=$!
  sleep 3
  t0=$(date +%s)
  curl -sS --max-time "${ROUND_TIMEOUT}" \
    -H "Authorization: Bearer ${API_KEY}" -H 'Content-Type: application/json' \
    "${BASE_URL}/v1/completions" \
    --data-binary "@${OUTDIR}/request_long.json" \
    >"${OUTDIR}/round${round}_long.json" 2>&1
  long_rc=$?
  elapsed=$(( $(date +%s) - t0 ))
  kill "${decode_pid}" >/dev/null 2>&1
  after=$(xid_count)

  prompt_tokens=$(python3 - "${OUTDIR}/round${round}_long.json" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    print(json.load(open(sys.argv[1]))["usage"]["prompt_tokens"])
except Exception:
    print(0)
PY
)
  if [ "${after}" -gt "${before}" ]; then
    log "round ${round}: FAIL — new Xid (${before} -> ${after})"
    verdict=1; break
  fi
  if [ "${long_rc}" -ne 0 ] || [ "${prompt_tokens}" -ne "${EXPECT_PROMPT_TOKENS}" ]; then
    log "round ${round}: FAIL — bad response (rc=${long_rc}, prompt_tokens=${prompt_tokens})"
    verdict=1; break
  fi
  if [ -n "${EXPECT_TEXT:-}" ] && ! grep -q "${EXPECT_TEXT}" "${OUTDIR}/round${round}_long.json"; then
    log "round ${round}: FAIL — expected '${EXPECT_TEXT}' missing"
    verdict=1; break
  fi
  log "round ${round}: pass (${elapsed}s, prompt_tokens=${prompt_tokens}, no Xid)"
done

# Greedy determinism across rounds doubles as a state-reuse check: the resume
# path must not change the continuation.
python3 - "${OUTDIR}" <<'PY'
import hashlib, json, pathlib, sys
d = pathlib.Path(sys.argv[1])
h = {}
for f in sorted(d.glob("round*_long.json")):
    try:
        t = json.load(open(f))["choices"][0]["text"]
    except Exception:
        continue
    h[f.name] = (len(t), hashlib.sha256(t.encode()).hexdigest()[:16])
for k, (n, digest) in h.items():
    print(f"  {k}: len={n} sha={digest}")
if len(h) > 1:
    print("  continuation identical across rounds:", len({v[1] for v in h.values()}) == 1)
PY

[ "${verdict}" -eq 0 ] && log "RESULT: PASS — ${ROUNDS} rounds clean" || log "RESULT: FAIL — see ${OUTDIR}"
exit "${verdict}"

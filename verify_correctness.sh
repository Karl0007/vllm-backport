#!/usr/bin/env bash
# Strict correctness battery for GLM-5.3-Flash (temperature 0).
# Every check has a deterministic expected answer; run against a booted server.
# Usage: ./verify_correctness.sh [PORT]
set -uo pipefail
PORT="${1:-8007}"
BASE="http://localhost:${PORT}"
MODEL="${MODEL_NAME:-glm-5.3-flash-nvfp4}"
PASS=0; FAIL=0

ask() {  # ask <json-escaped-user-msg> -> prints content only
  local msg="$1"
  curl -s "${BASE}/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": ${msg}}],
    \"temperature\": 0, \"max_tokens\": 1024,
    \"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}
  }" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message'].get('content') or '')"
}

check() {  # check <name> <msg> <expected-substring>
  local name="$1" msg="$2" expect="$3" out
  out=$(ask "$msg")
  if echo "$out" | grep -qF "$expect"; then echo "PASS  $name  (got: $(echo "$out" | tr '\n' ' ' | cut -c1-80))"; PASS=$((PASS+1));
  else echo "FAIL  $name  expected[$expect] got[$(echo "$out" | tr '\n' ' ' | cut -c1-200)]"; FAIL=$((FAIL+1)); fi
}

check "math-mult"    '"What is 13*17? Answer with just the number."' "221"
check "math-word"    '"What is one hundred and ninety-six divided by four? Answer with just the number."' "49"
check "capital"      '"What is the capital of France? One word."' "Paris"
check "reverse"      '"Reverse the string abcdef. Output only the reversed string."' "fedcba"
check "json-fn"      '"Return exactly this JSON and nothing else: {\"a\": [1, 2, 3], \"b\": null}"' '"b": null'
check "logic"        '"If all bloops are razzles and all razzles make noise, can we conclude a bloop makes noise? Answer yes or no first."' "Yes"
check "copy-long"    '"Repeat exactly, with nothing else: The quick brown fox jumps over the lazy dog 12345!"' "lazy dog 12345!"

# determinism: same greedy prompt twice must match
d1=$(ask '"Say exactly one color name, nothing else."')
d2=$(ask '"Say exactly one color name, nothing else."')
if [[ "$d1" == "$d2" && -n "$d1" ]]; then echo "PASS  determinism  ($d1)"; PASS=$((PASS+1));
else echo "FAIL  determinism  [$d1] vs [$d2]"; FAIL=$((FAIL+1)); fi

# tool call round-trip
curl -s "${BASE}/v1/chat/completions" -H "Content-Type: application/json" -d "{
  \"model\": \"${MODEL}\", \"temperature\": 0, \"max_tokens\": 512,
  \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in Shanghai? Use the get_weather tool.\"}],
  \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]
}" | python3 -c "
import sys, json
r = json.load(sys.stdin)
tc = (r['choices'][0]['message'].get('tool_calls') or [])
ok = tc and tc[0]['function']['name']=='get_weather' and 'shanghai' in tc[0]['function']['arguments'].lower()
print('PASS  tool-call' if ok else f'FAIL  tool-call  {tc}')
sys.exit(0 if ok else 1)" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo "== $PASS passed, $FAIL failed =="
[[ $FAIL -eq 0 ]]

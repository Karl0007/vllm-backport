#!/usr/bin/env python3
"""Long-context smoke test for qwen3.8-flash-next at 512K ctx + YaRN factor 2.

Places a needle early in a ~N-token filler haystack and asks for it at the end.
Verifies: (a) prompt > native 262144 is accepted, (b) rope tables/draft path
survive positions beyond native, (c) recall still works.
"""
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8010/v1/chat/completions"
KEY = "sk-cmp170hx"

FILLER = (
    "运维记录:第 {i} 节,节点 cmp170hx 的 rank{j} 在批次 {b} 完成了一次例行心跳自检,"
    "指标 KV 使用率 {p}%,解码吞吐 {q} tok/s,前缀缓存命中率 {r}%,状态正常无需干预。\n"
)


def post(messages, max_tokens=96, temperature=0.0):
    body = json.dumps(
        {
            "model": "qwen3.8-flash-next",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d, time.time() - t0


def main():
    blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 1400
    # Unique preamble: without this, a second run shares its whole prefix with the
    # first and the "probe" becomes a prefix-cache hit (a fake prefill rate).
    preamble = f"[run {random.randint(0, 2**31 - 1)}] 以下是一批运维记录。\n"
    needle = "工单号 ZQ-7731 的负责人是 骆冰 且复核码为 KX4-91-Δ"
    parts = [preamble]
    for i in range(blocks):
        if i == 3:  # needle early in the haystack
            parts.append(f"【重要】{needle}\n")
        parts.append(FILLER.format(i=i, j=i % 4, b=i * 7 % 997, p=i % 100, q=50 + i % 70, r=80 + i % 20))
    parts.append(
        "\n请在上面的运维记录里找到标记为【重要】的那一行,原样复述其中的工单号、负责人和复核码。"
        "如果找不到,只回答 NOT_FOUND。"
    )
    prompt = "".join(parts)

    # token-count probe: 1-token max_tokens, same prompt (prefix cache makes it cheap)
    probe, pt = post([{"role": "user", "content": prompt}], max_tokens=1)
    n = probe["usage"]["prompt_tokens"]
    print(
        f"PROBE prompt_tokens={n} native_limit=262144 beyond_native={n > 262144} "
        f"in_512k={n <= 524288} t={pt:.1f}s cold_prefill_tps={n / pt:.0f}"
    )

    d, el = post([{"role": "user", "content": prompt}], max_tokens=640)
    msg = d["choices"][0]["message"]
    txt = (msg.get("content") or "") + "\n" + (msg.get("reasoning") or "")
    u = d["usage"]
    hit = all(k in txt for k in ("ZQ-7731", "骆冰", "KX4-91-Δ"))
    # Second pass over the same prompt = prefix-cache hit, so this wall is decode
    # time, NOT prefill. Never read a prefill rate off it.
    print(
        f"RESULT prompt={u['prompt_tokens']} completion={u['completion_tokens']} "
        f"cached_wall={el:.1f}s decode_tps_approx={u['completion_tokens'] / el:.1f} "
        f"needle_hit={hit}"
    )
    print("ANSWER:", txt.strip()[:300].replace("\n", " "))


if __name__ == "__main__":
    main()

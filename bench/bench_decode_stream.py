#!/usr/bin/env python3
"""Decode throughput vs context, measured by STREAMING.

Upstream harness: allover326/deepseek-v4-cmp170hx bench/bench_decode_stream.py
(the source of the 98 t/s figure in its RESULTS.md), with URL/MODEL/auth env
overrides so the same numbers are comparable across our fork and old lines.

Streaming avoids the subtract-two-calls assumption: timestamp first token and
last, divide by the tokens in between. TTFT comes out of the same run.
"""
import json
import os
import random
import sys
import time
import urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8888/v1/completions")
MODEL = os.environ.get("BENCH_MODEL", "deepseek-v4-flash-0731")
_KEY = os.environ.get("BENCH_KEY", "")
HDRS = {"Content-Type": "application/json"}
if _KEY:
    HDRS["Authorization"] = f"Bearer {_KEY}"
N_DECODE = 192

WORDS = (
    "system kernel memory buffer thread process socket packet register cache "
    "pointer allocate schedule interrupt virtual physical address translate "
    "compile execute branch predict pipeline vector matrix tensor gradient "
    "cluster network storage device driver module segment offset boundary"
).split()


def make_prompt(approx_tokens, seed):
    rng = random.Random(seed)
    return f"[run {seed}] " + " ".join(
        rng.choice(WORDS) for _ in range(int(approx_tokens / 1.3))
    )


def stream_once(prompt, max_tokens, timeout=3600):
    req = urllib.request.Request(
        URL,
        json.dumps(
            {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
                # Force exactly max_tokens: without this DSpark diverges and
                # hits EOS early, so comparisons read different lengths.
                "ignore_eos": True,
                # ★ Under speculative decoding one SSE chunk carries several
                # tokens; count the server's completion_tokens, not chunks.
                "stream_options": {"include_usage": True},
            }
        ).encode(),
        HDRS,
    )
    t0 = time.perf_counter()
    t_first = None
    n_chunks = 0
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                n_tokens = obj["usage"].get("completion_tokens", n_tokens)
            if not obj.get("choices"):
                continue
            if obj["choices"][0].get("text"):
                if t_first is None:
                    t_first = time.perf_counter()
                n_chunks += 1
    t_end = time.perf_counter()
    ttft = (t_first - t0) if t_first else float("nan")
    dec_s = (t_end - t_first) if t_first else float("nan")
    n = n_tokens or n_chunks
    # The first token came out of prefill, so it is not a decode step.
    rate = (n - 1) / dec_s if t_first and dec_s > 0 and n > 1 else float("nan")
    return ttft, dec_s, n, rate


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    targets = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [
        2000, 8000, 32000, 65000, 100000
    ]
    print(f"########## {label} :: DECODE vs CONTEXT (streaming) ##########")
    stream_once(make_prompt(2000, 1), 8)  # warm-up discard
    print(f"{'ctx tok':>9} {'TTFT s':>8} {'decode s':>9} {'gen':>5} {'decode t/s':>11}")
    for i, t in enumerate(targets):
        try:
            ttft, dec_s, n, rate = stream_once(make_prompt(t, 4242 + i), N_DECODE)
            approx_ctx = int(t / 1.3 * 1.3)
            print(f"{approx_ctx:>9} {ttft:>8.2f} {dec_s:>9.2f} {n:>5} {rate:>11.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"{t:>9}  FAILED: {str(e)[:110]}")


if __name__ == "__main__":
    main()

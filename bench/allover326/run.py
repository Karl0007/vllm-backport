#!/usr/bin/env python3
"""Runner shim for the vendored allover326 bench harnesses (byte-identical
copies kept in this directory; see README.upstream.md for their semantics).

Those scripts predate our key-guarded ports (plain headers; some bake
URL/MODEL constants). Inject everything from here, keep vendored files pristine:

  BENCH_KEY    -> Bearer header on every urlopen aimed at localhost/127.0.0.1
  BENCH_MODEL  -> substituted into any JSON body that carries "model"

Only urllib.request.urlopen is wrapped. Do NOT rebind urllib.request.Request:
it is a factory function whose body does isinstance(url, Request) against the
MODULE GLOBAL, so rebinding it makes every request raise TypeError.

Usage:
  BENCH_KEY=sk-... [BENCH_MODEL=glm-5.3-flash-awq] \
    python3 run.py bench_concurrency.py LABEL
  BENCH_KEY=sk-... python3 run.py bench_needle.py --port 8889 \
        --model glm-5.3-flash-awq        # scripts that take real args
"""
import json, os, runpy, sys, urllib.request

_KEY = os.environ.get("BENCH_KEY", "")
_MODEL = os.environ.get("BENCH_MODEL", "")
_PORT = os.environ.get("BENCH_PORT", "")  # retarget hardcoded localhost:PORT URLs
_orig_open = urllib.request.urlopen

def _is_request(obj):
    return hasattr(obj, "full_url") and hasattr(obj, "add_header") and hasattr(obj, "data")

def _open(req, *a, **kw):
    if _is_request(req):
        local = "127.0.0.1" in req.full_url or "localhost" in req.full_url
        if _PORT and local:
            import re
            req.full_url = re.sub(r"(127\.0\.0\.1|localhost):\d+",
                                  rf"\g<1>:{_PORT}", req.full_url)
            local = True
        if _KEY and local and "Authorization" not in req.headers:
            req.add_header("Authorization", f"Bearer {_KEY}")
        if _MODEL and isinstance(req.data, (bytes, bytearray)) and b'"model"' in req.data:
            try:
                obj = json.loads(req.data)
                if isinstance(obj, dict) and obj.get("model"):
                    obj["model"] = _MODEL
                    req.data = json.dumps(obj).encode()
                    # stale Content-Length (if the script set one) -> let
                    # urllib recompute it from data
                    req.headers.pop("Content-length", None)
                    req.unredirected_hdrs.pop("Content-length", None)
            except Exception:
                pass
    return _orig_open(req, *a, **kw)

urllib.request.urlopen = _open

if len(sys.argv) < 2:
    sys.exit(__doc__)
_script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), _script),
               run_name="__main__")

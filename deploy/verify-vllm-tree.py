#!/usr/bin/env python3
"""Assert the upstream fixes this runtime image is required to carry.

Two things were wrong with the version this replaces, and both are worth
keeping in mind when editing it:

* it lived in a ``RUN python3 - <<'EOF'`` block, and the builder in use here
  does not expand heredocs -- the step ran an empty script and reported
  success, so nothing below was ever enforced;
* the vllm#53142 entry asserted the presence of ``self._mamba_block_size``,
  which is exactly the upstream text that carries the bug. The check ratified
  the defect it was supposed to catch.

So: assertions read source text only (no module imports -- the build container
has no GPU, and importing vllm fails there), and the mamba seed check now
requires the *fixed* form. The locally patched files are asserted by
apply-vllm-patches.py, which runs before this one.
"""

from __future__ import annotations

import glob
import os
import sys


def find_root() -> str:
    for pattern in (
        "/usr/local/lib/python3*/dist-packages/vllm/__init__.py",
        "/usr/local/lib/python3*/site-packages/vllm/__init__.py",
    ):
        for path in sorted(glob.glob(pattern)):
            return os.path.dirname(path)
    sys.exit("no installed vllm package found")


ROOT = find_root()


def read(*parts: str) -> str:
    path = os.path.join(ROOT, *parts)
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    return open(path).read()


def require(text: str, marker: str, why: str) -> None:
    if marker not in text:
        sys.exit(f"{why} (missing {marker!r})")


def forbid(text: str, marker: str, why: str) -> None:
    if marker in text:
        sys.exit(f"{why} (found {marker!r})")


# PR #63: the align copy kernels index the SOURCE req-indexed tables by req_idx,
# and the align context is bound to those source tables.
mamba_utils = read("v1", "worker", "mamba_utils.py")
require(mamba_utils, "bt_row_idx = req_idx", "mamba postprocess indexes batch rows")
require(
    mamba_utils,
    "Source tables are req-indexed",
    "mamba align pre-copy indexes batch rows",
)
forbid(
    mamba_utils,
    "HAS_IDX_MAPPING else req_idx",
    "stale batch-row indexing is back",
)
require(
    read("v1", "worker", "gpu", "model_states", "mamba_hybrid.py"),
    "SOURCE per-request-slot tables",
    "mamba align ctx not bound to the source block tables",
)

# vllm#53142 (the form this branch needs): the state column divisor is the
# resolved mamba spec block size, not the CLI/cache block size.
mamba_hybrid = read("v1", "worker", "gpu", "model_states", "mamba_hybrid.py")
require(
    mamba_hybrid,
    "self._mamba_block_size = int(mamba_spec.block_size)",
    "align state column divisor is not the resolved mamba spec block size",
)
require(
    mamba_hybrid,
    "_pending_state_seed",
    "resume positions parked by add_request are never seeded",
)

# #55223: reasoning-end offset scan.
require(
    read("v1", "structured_output", "__init__.py"),
    "find_reasoning_end_offset",
    "#55223 reasoning-offset scan missing",
)

# GLM PP hand-off and the MTP draft embedding path.
require(
    read("models", "glm5next", "nvidia", "model.py"),
    "mhc",
    "GLM PP mHC hand-off missing",
)
require(
    read("v1", "worker", "gpu", "spec_decode", "eagle", "utils.py"),
    "maybe_share_target_embed",
    "MTP draft embedding wiring missing",
)

# Qwen UVA PLE offload replaces the PleOffloadWorker design.
ngram_embedding = read("models", "qwen4_exp", "nvidia", "ngram_embedding.py")
require(
    ngram_embedding,
    "Qwen4ExpPLEPinnedHostEmbedding",
    "UVA PLE offload missing",
)
require(ngram_embedding, "is_uva_available", "UVA guard missing")

# Deliberate omission: fail if vllm#48375's extra block drop is re-ported
# (upstream rejects it, a47f0f82d5).
forbid(
    read("v1", "core", "single_type_kv_cache_manager.py"),
    "max_length = max(0, max_length - kv_cache_spec.block_size)",
    "re-ported #48375 extra block drop",
)

print("overlay assertions OK")

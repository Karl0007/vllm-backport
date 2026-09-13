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


def require_count(text: str, marker: str, count: int, why: str) -> None:
    found = text.count(marker)
    if found != count:
        sys.exit(f"{why} (expected {count}x {marker!r}, found {found})")


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

# upstream #56362 (unmerged): the pinned lookup is a byte copy, so FP8 tables
# are passed as uint8 -- otherwise Triton rejects the fp8e4nv pointer on SM8x
# and the engine dies in profile_run.
require(
    ngram_embedding,
    "weight_arg = weight_arg.view(torch.uint8)",
    "PLE UVA lookup passes an fp8e4nv pointer to Triton again (SM8x startup failure)",
)
require(
    ngram_embedding,
    "output_arg = output_arg.view(torch.uint8)",
    "PLE UVA lookup output is not byte-viewed for FP8 tables",
)

# upstream #54793 (unmerged): a KV cache group that projects to empty on a PP
# rank must not emit tensors for other ranks' layers -- allocate_kv_cache then
# dies with a bare StopIteration during engine init.
require(
    read("v1", "core", "kv_cache_utils.py"),
    "for layer_name, spec in group_spec.kv_cache_specs.items():\n"
    "                if layer_name in group.layer_names:\n"
    "                    layers_by_spec[spec].append(layer_name)",
    "empty projected KV-cache groups emit other ranks' tensors (PP StopIteration)",
)

# upstream #55390 (unmerged): the positional MTP draft-group fallback must
# apply to every MTP model (not just deepseek_v4) on both grouping paths,
# else the flag-all fallback silently disables prefix-cache reuse for the
# Mamba/GDN groups.
_kv_cache_utils = read("v1", "core", "kv_cache_utils.py")
require(
    _kv_cache_utils,
    "_groups_partition_layers_exactly(kv_cache_spec, kv_cache_groups)",
    "positional draft-group fallback missing its exact-partition guard",
)
require_count(
    _kv_cache_utils,
    "use_trailing_layer_fallback=_uses_trailing_mtp_layers(vllm_config)",
    2,
    "MTP draft groups not annotated on every grouping path (prefix reuse disabled)",
)

# Deliberate omission: fail if vllm#48375's extra block drop is re-ported
# (upstream rejects it, a47f0f82d5).
forbid(
    read("v1", "core", "single_type_kv_cache_manager.py"),
    "max_length = max(0, max_length - kv_cache_spec.block_size)",
    "re-ported #48375 extra block drop",
)

print("overlay assertions OK")

# ---------------------------------------------------------------------------
# split-KV spec-decode port (x99 line). One semantic marker per registered file:
# a string that can only be present when the fix itself is present, never a name.
# ---------------------------------------------------------------------------
spec_attn = read("v1", "attention", "ops", "spec_decode_attn.py")
require(
    spec_attn,
    "k_lim = nblocks * stride_kb",
    "composed KV-gather address is unbounded again: a consistent-but-stale parameter "
    "set walks out of the cache (Xid 31, negative offset)",
)
require(
    spec_attn,
    "g_ok = k_pos[:, None] & (k_off >= 0) & (k_off < k_lim)",
    "KV gather lost its composed-offset mask",
)
require_count(
    spec_attn,
    "_spec_attn_partial[grid]",
    1,
    "split-KV verify kernel launch missing or duplicated",
)

fa2 = read("v1", "attention", "backends", "flash_attn.py")
require(fa2, "def _spec_attn_run(", "FA2 backend lost the split-KV verify entry point")
require(fa2, "VLLM_SPEC_DECODE_ATTN", "split-KV env switch gone (now unconditional or dead)")

fi = read("v1", "attention", "backends", "flashinfer.py")
require(
    fi,
    "prefill_real_tokens",
    "FlashInfer q/qo_indptr mismatch is back: fp8 KV plus speculation dies at startup",
)

eagle = read("v1", "worker", "gpu", "spec_decode", "eagle", "utils.py")
require(eagle, "PPMissingLayer", "draft embed lookup aliases a missing layer again")

rs = read("v1", "sample", "rejection_sampler.py")
require(rs, "num_draft_tokens", "rejection sampler lost its spec-decode path")

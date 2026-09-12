#!/usr/bin/env python3
"""Assert the upstream fixes this runtime image is required to carry.

Extracted verbatim from a ``RUN python3 - <<'EOF'`` block in
Dockerfile.cmp170hx. The legacy builder does not expand heredocs, so that
block ran *nothing* and printed success, which is why the port's checklist
(including the wrong "seed divisor is already upstream" entry) was never
enforced. As a real file the step either passes or fails the build.
"""

import importlib, os, sys
mods = [
    "vllm.models.glm5next.nvidia.model",
    "vllm.models.qwen4_exp.nvidia.model_state",
    "vllm.models.qwen4_exp.nvidia.ngram_embedding",
    "vllm.v1.worker.gpu.spec_decode.eagle.utils",
    "vllm.v1.core.sched.async_scheduler",
]
for m in mods:
    importlib.import_module(m)

# Text assertions (triton JIT objects / plain scripts): the fixes this line
# is required to carry upstream.
root = os.path.dirname(importlib.import_module("vllm").__file__)
read = lambda *p: open(os.path.join(root, *p)).read()

# PR #63: kernels index the SOURCE req-indexed tables by req_idx.
mu = read("v1", "worker", "mamba_utils.py")
assert "bt_row_idx = req_idx" in mu, "mamba postprocess still indexes batch rows"
assert "HAS_IDX_MAPPING else req_idx" not in mu, "stale batch-row indexing is back"
assert "Source tables are req-indexed" in mu, "mamba precopy still indexes batch rows"

# PR #63 runner half: ctx bound to source tables in preprocess_state.
mr = read("v1", "worker", "gpu", "model_runner.py")
assert "tuple(bt.gpu for bt in self.block_tables.block_tables)" in mr, (
    "mamba ctx not bound to source block tables"
)

# vllm#53142: state-seed divisor is the mamba group block size.
mh = read("v1", "worker", "gpu", "model_states", "mamba_hybrid.py")
assert "self._mamba_block_size" in mh, "mamba state seed uses CLI block size divisor"

# #55223: reasoning-end offset scan present.
so = read("v1", "structured_output", "__init__.py")
assert "find_reasoning_end_offset" in so, "#55223 reasoning-offset scan missing"

# GLM PP hand-off + MTP embed-from-checkpoint.
gm = read("models", "glm5next", "nvidia", "model.py")
assert "mHC" in gm or "mhc" in gm, "GLM PP mHC hand-off missing"
mt = read("models", "glm5next", "nvidia", "mtp.py")
assert "PPMissingLayer" in mt or "pp_missing" in mt.lower(), (
    "MTP draft embed-from-checkpoint (PP) missing"
)

# Qwen UVA PLE offload replaces the PleOffloadWorker design.
ne = read("models", "qwen4_exp", "nvidia", "ngram_embedding.py")
assert "Qwen4ExpPLEPinnedHostEmbedding" in ne, "UVA PLE offload missing"
assert "is_uva_available" in ne, "UVA guard missing"

# Deliberate omissions (fail the build if someone re-ports them):
st = read("v1", "core", "single_type_kv_cache_manager.py")
assert "max_length = max(0, max_length - kv_cache_spec.block_size)" not in st, (
    "re-ported #48375 extra block drop upstream rejects (a47f0f82d5)"
)
print("overlay assertions OK")

#!/usr/bin/env python3
"""Apply this branch's local vllm/ patches to the image's installed package.

The Dockerfile used to carry these steps as ``RUN python3 - <<'EOF'`` blocks.
The legacy builder in use here does not expand heredocs: it runs an empty
script, exits 0 and reports success, so every such block in this file was a
no-op -- including the port-time "assertions pin the fixes" check. That is how
a wrong checklist entry (the align-mode seed divisor) shipped. This script is a
real file so the step either runs or the build fails.

Run as: ``python3 deploy/apply-vllm-patches.py <stage-dir>``
"""

from __future__ import annotations

import glob
import os
import shutil
import sys

PATCHES = {
    "mamba_hybrid.py": "v1/worker/gpu/model_states/mamba_hybrid.py",
    "fla_index.py": "third_party/flash_linear_attention/ops/index.py",
    "ngram_embedding.py": "models/qwen4_exp/nvidia/ngram_embedding.py",
    "kv_cache_utils.py": "v1/core/kv_cache_utils.py",
}

# Split-KV spec-decode port (x99 line). Same rule as above: explicit list, one
# semantic REQUIRED entry per file below, no whole-tree copy. Stage names carry a
# prefix because several of these files share a basename with an upstream module
# (core.py, utils.py, scheduler.py, interface.py) and the stage dir is flat.
PATCHES.update({
    "spec_decode_attn.py": "v1/attention/ops/spec_decode_attn.py",
    "flash_attn_be.py": "v1/attention/backends/flash_attn.py",
    "flashinfer_be.py": "v1/attention/backends/flashinfer.py",
    "engine_core.py": "v1/engine/core.py",
    "sched_interface.py": "v1/core/sched/interface.py",
    "sched_scheduler.py": "v1/core/sched/scheduler.py",
    "eagle_utils.py": "v1/worker/gpu/spec_decode/eagle/utils.py",
    "reject_sampler.py": "v1/sample/rejection_sampler.py",
    "warmup.py": "v1/worker/gpu/warmup.py",
    "qwen_dflash.py": "model_executor/models/qwen3_dflash.py",
    "qwen3_5.py": "model_executor/models/qwen3_5.py",
    "qwen3_5_mtp.py": "model_executor/models/qwen3_5_mtp.py",
})

REQUIRED = (
    (
        "v1/worker/gpu/model_states/mamba_hybrid.py",
        "self._mamba_block_size = int(mamba_spec.block_size)",
        "align state column divisor must come from the resolved mamba spec",
    ),
    (
        "v1/worker/gpu/model_states/mamba_hybrid.py",
        "_pending_state_seed",
        "deferred resume seeding missing",
    ),
    (
        "models/qwen4_exp/nvidia/ngram_embedding.py",
        "weight_arg = weight_arg.view(torch.uint8)",
        "PLE UVA lookup hands Triton an fp8e4nv pointer again (SM8x dies in profile_run)",
    ),
    (
        "v1/core/kv_cache_utils.py",
        "for layer_name, spec in group_spec.kv_cache_specs.items():\n"
        "                if layer_name in group.layer_names:\n"
        "                    layers_by_spec[spec].append(layer_name)",
        "empty projected KV-cache groups emit other ranks' tensors again (PP StopIteration)",
    ),
    (
        "v1/core/kv_cache_utils.py",
        "use_trailing_layer_fallback=_uses_trailing_mtp_layers(vllm_config)",
        "MTP draft KV groups are unannotated again (flag-all disables prefix reuse)",
    ),
    # --- split-KV spec-decode port -------------------------------------------------
    (
        "v1/attention/ops/spec_decode_attn.py",
        "k_lim = nblocks * stride_kb",
        "composed KV-gather address is unbounded again: a stale-but-consistent "
        "parameter set walks out of the cache (Xid 31)",
    ),
    (
        "v1/attention/ops/spec_decode_attn.py",
        "_spec_attn_partial[grid]",
        "split-KV verify kernel missing",
    ),
    (
        "v1/attention/backends/flash_attn.py",
        "def _spec_attn_run(",
        "FA2 backend no longer routes multi-token verify through the split-KV hook",
    ),
    (
        "v1/attention/backends/flashinfer.py",
        "prefill_real_tokens",
        "FlashInfer q/qo_indptr mismatch is back: fp8 KV plus speculation dies at "
        "startup with 'q.shape[0] (16) does not match qo_indptr[-1] (8)'",
    ),
    (
        "v1/engine/core.py",
        "def _merge_engine_core_outputs(",
        "engine core wiring for the split-KV port missing",
    ),
    (
        "v1/sample/rejection_sampler.py",
        "if num_draft_tokens < 0 or num_draft_tokens > max_spec_len:",
        "rejection sampler lost its spec-decode bounds path",
    ),
    (
        "v1/worker/gpu/spec_decode/eagle/utils.py",
        "_EMBED_KEYS = (",
        "draft embed lookup aliases a missing layer instead of skipping it",
    ),
    (
        "v1/core/sched/interface.py",
        "def has_structured_output_in_flight(",
        "scheduler interface hook for the split-KV port missing",
    ),
    (
        "v1/core/sched/scheduler.py",
        "def has_structured_output_in_flight(",
        "scheduler implementation of the split-KV in-flight hook missing",
    ),
    (
        "v1/worker/gpu/warmup.py",
        "VLLM_SKIP_WARMUP",
        "warmup switch for quantized KV + speculation gone (or renamed)",
    ),
    (
        "model_executor/models/qwen3_dflash.py",
        "def _can_fuse_context_kv(",
        "DFlash context-KV fusion entry point missing",
    ),
    (
        "model_executor/models/qwen3_5.py",
        'prefix=maybe_prefix(prefix, "embed_tokens"),',
        "Qwen3.5 embed_tokens prefix fix missing (draft embed aliasing)",
    ),
    (
        "model_executor/models/qwen3_5_mtp.py",
        'prefix=maybe_prefix(prefix, "embed_tokens"),',
        "Qwen3.5-MTP embed_tokens prefix fix missing (draft embed aliasing)",
    ),
)


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vllm_patch"

    # The build-stage interpreter and the runtime console script do not
    # necessarily resolve `vllm` to the same tree, so patch every candidate
    # install root rather than guessing one.
    roots = sorted(
        {
            os.path.dirname(p)
            for pattern in (
                "/usr/local/lib/python3*/dist-packages/vllm/__init__.py",
                "/usr/local/lib/python3*/site-packages/vllm/__init__.py",
                "/usr/lib/python3*/dist-packages/vllm/__init__.py",
                "/usr/lib/python3*/site-packages/vllm/__init__.py",
            )
            for p in glob.glob(pattern)
        }
        | set(glob.glob("/usr/local/lib/python3*/dist-packages/vllm"))
        | set(glob.glob("/usr/local/lib/python3*/site-packages/vllm"))
    )
    if not roots:
        print("no installed vllm package found under any python tree", file=sys.stderr)
        return 1

    patched = 0
    for root in roots:
        targets = {name: os.path.join(root, rel) for name, rel in PATCHES.items()}
        # A port may add files the base tree does not have (spec_decode_attn.py),
        # so "every target already exists" is the wrong gate -- it made the whole
        # apply step a no-op and failed the build with "no candidate tree
        # contained the patch targets". Require at least one existing target to
        # recognise a real install root, then let the copy create the rest.
        if not any(os.path.exists(path) for path in targets.values()):
            continue
        created: list[str] = []
        for name, rel in PATCHES.items():
            src = os.path.join(stage, name)
            if os.path.exists(src):
                dst = os.path.join(root, rel)
                if not os.path.exists(dst):
                    created.append(rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

        for rel, marker, why in REQUIRED:
            text = open(os.path.join(root, rel)).read()
            if marker not in text:
                print(f"{root}: {why} (missing {marker!r})", file=sys.stderr)
                return 1
        index_text = open(targets["fla_index.py"]).read()
        if "tensor_cache" in index_text:
            print(
                f"{root}: per-step chunk metadata is cached by identity again",
                file=sys.stderr,
            )
            return 1
        patched += 1
        suffix = f" (created {len(created)}: {', '.join(created)})" if created else ""
        print(f"patched {root}{suffix}")

    if not patched:
        print("no candidate tree contained the patch targets", file=sys.stderr)
        return 1
    print(f"local vllm/ patches applied to {patched} tree(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

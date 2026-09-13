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
        if not all(os.path.exists(path) for path in targets.values()):
            continue
        for name, rel in PATCHES.items():
            src = os.path.join(stage, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(root, rel))

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
        print(f"patched {root}")

    if not patched:
        print("no candidate tree contained the patch targets", file=sys.stderr)
        return 1
    print(f"local vllm/ patches applied to {patched} tree(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

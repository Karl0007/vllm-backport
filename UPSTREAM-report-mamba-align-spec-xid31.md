# Upstream report draft — hybrid Mamba/GDN + spec decode: Xid 31 in the align mamba path

Status: open. Filled in by the local session on 2026-09-13. Ready to file.
Everything below is measured on the hardware in `../BENCH-27b-kkx99.md`; see that
file for the surrounding performance work.

## Environment

- Model: Qwen3.8-27B (hybrid: 48 GDN linear-attention layers + 16 full-attention
  layers), AWQ int4 with eq8emb embeddings, max_model_len 262144.
- Draft: DFlash2-W4A16, `num_speculative_tokens=7`.
- Engine: this fork's backport image `vllm/vllm-backport:cmp170hx-v013-nopatch`,
  base `origin/master` at tag v0.13.0 (2026-09-11), V2 GPU model runner.
- Flags: `--enable-prefix-caching --mamba-cache-mode align --max-num-seqs 8
  --max-num-batched-tokens 8192`, async scheduling on, CUDA graphs
  (`FULL_AND_PIECEWISE`, the default; PIECEWISE alone behaves the same).
- GPU: CMP 170HX (GA100, sm80), driver 610.43.02, single card.

## Symptom

`NVRM: Xid (PCI:...): 31 ... MMU Fault: ENGINE GRAPHICS ... FAULT_PDE
ACCESS_TYPE_VIRT_READ faulted @ 0x25_e...` — the engine dies; with async scheduling
the Python error surfaces later ("CUDA error: an illegal memory access") with no
kernel attribution. The fault address is consistently **below** the largest CUDA
segment: `torch.cuda.memory_snapshot()` puts KV cache + mamba state in one
`0x26e0000000..0x3019400000` (37.8 GiB) segment, and every observed fault address
(0x25e0278000, 0x25e0581000, 0x25e08c0000, 0x25e0240000, 0x25e4680000, 0x25e6d80000)
sits *under* it — i.e. the read used a negative offset from the state/KV base.

With `CUDA_LAUNCH_BLOCKING=1` and `--enforce-eager`, two earlier faults of this
family pinned to `vllm/v1/worker/gpu/model_states/mamba_hybrid.py:preprocess_state`
-> `vllm/v1/worker/mamba_utils.py:run_fused_precopy`
(`precopy_mamba_align_fused_kernel`). Those two had concrete causes in this tree and
are fixed (see "Fixes already applied"); the residual below is what is left.

## Reproduction

Single server, single stream, CUDA graphs on:

```
POST /v1/chat/completions  ~248000 chars   (≈135k tokens)   # parent prompt
POST /v1/chat/completions   ~96000 chars   (≈52k tokens)    # a prefix of the parent
POST /v1/chat/completions   ~96000 chars
POST /v1/chat/completions   ~60000 chars
```

Loop those four; the engine dies within 1–4 rounds. The second and later requests
resume from a cached prefix (TTFT collapses to ~1 s), which is the state that
matters. It also fires with the parent/child pair repeated verbatim
(135k twice, 65k twice, …).

## Characterisation matrix (all measured on this box)

| Configuration | Result |
|---|---|
| CUDA graphs + spec decode + prefix-cache resume + long context | **faults** (~1 per 12–16 long requests; sometimes on the first) |
| `--enforce-eager` (rest identical) | clean |
| `SPEC=none` / no draft model (rest identical) | clean (4 rounds) |
| `SPEC_ATTN=0` — split-KV verify attention disabled | **still faults** (first request) |
| contexts below ~32k | never observed |
| any instrumentation (kernel `tl.device_print`, or a host sync in `preprocess_state`) | **moves the failure**: a tree that passed 12 requests faults on the first one |

That last row is the most useful signal: the defect is timing-sensitive, and adding
enough work changes when it fires.

## What is ruled out (each with evidence, not by inspection)

- **Out-of-range block-table columns** in `_copy_mamba_state_block`: guarded, print
  never fired in 24 faulting/clean requests.
- **Block-table reallocation** (stale `block_table_ptrs`): a host-side pointer
  invariant over `block_tables` never fired.
- **Wrong block size in the postprocess path**: `self.block_size` is
  `mamba_spec.block_size` (832 here) — correct.
- **The align preprocess column math**: kernel-side scalars at the failing step are
  sane (`pre_state_idx=63`, `num_computed=52578`, divisor 832, `n_accepted=1`).
- **The speculative verify attention** (our split-KV kernel): faults with it off.
- **RecoverSSM lazy state writes**: constructed only for `use_kda_recoverssm`
  (Kimi-K3 KDA), not active for this GDN model.
- **`num_accepted_tokens` snapshot ordering**: the snapshot is a same-stream copy in
  `run_fused_postprocess_align`, so it is ordered before the kernel that reads it.

## Fixes already applied on this tree (they removed two deterministic faults)

1. A carried port of vllm#48375 (MambaManager dropping one extra matched block on an
   EAGLE/spec-decode prefix hit) was removed: on this base upstream
   `get_replay_boundaries` (#53945/#54713) already retains the checkpoint, and the
   extra drop zeroes resends. Verified: 8-way harness + four identical 64.7k sends
   (cache hits) clean, 169 tok/s.
2. The worker seeded the mamba running-state column with
   `cache_config.mamba_block_size` (16 — the CLI default `--block-size`, latched
   before `platforms/interface.py` raises `cache_config.block_size` to the attention
   block size). The state table is dimensioned by the mamba group's block size
   (832), so the column was ~52x out of range. Fixed by seeding lazily, from
   `mamba_spec.block_size`, once the kv cache config is known (vllm#53142's
   "out-of-range block_table column" family, opposite direction). Verified: 135k
   after a 64.7k parent, four rounds clean.

## Suspect area for the residual

Everything left points at the spec-decode mamba *save/advance* path in the align
mode: `postprocess_mamba_fused_kernel` ("save the running state to the block-aligned
position after spec-decode acceptance leaves the sequence non-aligned") and its
interaction with the next step's `precopy_mamba_align_fused_kernel`, under CUDA-graph
replay. It only runs with speculation, it only matters when the sequence leaves a
block boundary through acceptance, and it is the only path that writes state columns
from an *accepted-token* position rather than a computed one.

Related upstream work in the same area: #53479 (materialize a state at every boundary
a lookup can reach; drop the speculative one-block back-off), #50409, #51113, #45477,
#52371.

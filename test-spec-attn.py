"""Numeric + regression check for the split-KV spec-decode attention kernel
(vllm/v1/attention/ops/spec_decode_attn.py, enabled with SPEC_ATTN=1).

Run (no model load, ~15 s, needs one free GPU):

  docker run --rm --gpus device=1 --ipc=host --entrypoint bash \
    -v "$PWD/vllm/v1/attention/ops/spec_decode_attn.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/spec_decode_attn.py:ro" \
    -v "$PWD/test-spec-attn.py:/t.py:ro" \
    vllm/vllm-backport:cmp170hx-v013-nopatch -c "cd /tmp && python3 /t.py"

Covers: GQA group sizes 2/6/8/16/32, head dims 64/128/256, the multi-tile query
path, KV lengths around tile/segment boundaries, and the padded-persistent-buffer
regression (stale cu_seqlens_q tail from a previous 8-request batch with zeroed
seqused_k pads) that crashed with an illegal address on 2026-09-12.

"""
"""Standalone numerical check of the ported split-KV spec-decode kernel.

Feeds the kernel a paged KV cache directly (no vLLM metadata) and compares every
output row against an eager reference that applies the documented masking:
query token i of a request sits at kv position kv_len - q_len + i and attends
causally up to and including that position.
"""
import math
import sys

import torch

from vllm.v1.attention.ops.spec_decode_attn import SpecDecodeAttention

DEV = "cuda"
BS = 16  # block size; the kernel requires a multiple of 16
QMAX = 8  # query tokens per request

# boundary-heavy lengths: < TILE, == TILE, ±1, multiple of BS, multi-segment
KV_LENS = [16, 17, 63, 64, 65, 129, 255, 256, 1024, 4096, 16384]


def reference(q, kcache, vcache, bt, kv_lens, q_lens, cu, Hq, Hkv, D, scale):
    G = Hq // Hkv
    ref = torch.empty_like(q, dtype=torch.float32)
    for r, (kl, ql) in enumerate(zip(kv_lens, q_lens)):
        nblocks = (kl + BS - 1) // BS
        ids = bt[r, :nblocks].long()
        K = kcache[ids].reshape(-1, Hkv, D)[:kl].float()  # [kl, Hkv, D]
        V = vcache[ids].reshape(-1, Hkv, D)[:kl].float()
        start = int(cu[r])
        for i in range(ql):
            pos = kl - ql + i  # last allowed position (inclusive)
            if pos < 0:
                ref[start + i] = 0.0
                continue
            for h in range(Hq):
                kh = h // G
                scores = (q[start + i, h].float() @ K[: pos + 1, kh].T) * scale
                p = torch.softmax(scores, dim=-1)
                ref[start + i, h] = p @ V[: pos + 1, kh]
    return ref


def run_case(Hq, Hkv, D, label):
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(D)
    q_lens = [min(QMAX, kl) for kl in KV_LENS]
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()),
                      dtype=torch.int32, device=DEV)

    total_blocks = sum((kl + BS - 1) // BS for kl in KV_LENS) + 4
    max_blocks = max((kl + BS - 1) // BS for kl in KV_LENS)
    kcache = torch.randn(total_blocks, BS, Hkv, D, dtype=torch.bfloat16, device=DEV) * 0.5
    vcache = torch.randn(total_blocks, BS, Hkv, D, dtype=torch.bfloat16, device=DEV) * 0.5

    bt = torch.zeros(len(KV_LENS), max_blocks, dtype=torch.int32, device=DEV)
    nxt = 0
    for r, kl in enumerate(KV_LENS):
        n = (kl + BS - 1) // BS
        bt[r, :n] = torch.arange(nxt, nxt + n, dtype=torch.int32, device=DEV)
        nxt += n
    seqused = torch.tensor(KV_LENS, dtype=torch.int32, device=DEV)

    T = int(cu[-1])
    q = (torch.randn(T, Hq, D, dtype=torch.float32, device=DEV) * 0.5).to(torch.bfloat16)
    out = torch.zeros_like(q)

    att = SpecDecodeAttention(len(KV_LENS), Hq, D, torch.device(DEV), QMAX)
    att.run(q, kcache, vcache, out, cu, seqused, bt, scale, len(KV_LENS), max(q_lens))

    ref = reference(q, kcache, vcache, bt, KV_LENS, q_lens, cu, Hq, Hkv, D, scale)
    got = out.float()

    # Reference rows that attend a single position are exact; near-empty rows leak
    # little. Compare absolutely and relative to the row norm.
    err = (got - ref).abs()
    denom = ref.abs().amax(dim=-1, keepdim=True).clamp_min(1e-3)
    rel = (err / denom).max().item()
    max_abs = err.max().item()
    print(f"{label}: Hq={Hq} Hkv={Hkv} D={D} rows={T} max_abs_err={max_abs:.4e} max_rel={rel:.4e}")
    return max_abs, rel


ok = True
for Hq, Hkv, D, label in [
    (32, 4, 256, "ours(G=8,D=256)"),
    (8, 4, 128, "small-D(G=2)"),
    (128, 4, 128, "multi-tile(G=32)"),
    (32, 2, 64, "D=64(G=16)"),
]:
    max_abs, rel = run_case(Hq, Hkv, D, label)
    if max_abs > 2e-2 and rel > 2e-2:
        ok = False
        print(f"  ^^ FAIL: {label}")

# ---- regression: padded persistent buffers (the crash we observed) -------------
# cu_seqlens_q keeps the previous 8-request batch's prefix sums in its tail while
# seqused_k pads are zero. Only request 0 is real; the kernel must not touch rows
# beyond this step's 8 tokens.
def run_stale_tail_case(Hq=24, Hkv=4, D=256):
    torch.manual_seed(1)
    scale = 1.0 / math.sqrt(D)
    qlen_real, kv_real = 8, 80690
    pad_reqs = 8
    cu = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56, 64], dtype=torch.int32, device=DEV)[: pad_reqs + 1]
    cu[1] = qlen_real
    seqused = torch.zeros(pad_reqs, dtype=torch.int32, device=DEV)
    seqused[0] = kv_real

    nblocks = (kv_real + BS - 1) // BS
    kcache = torch.randn(nblocks, BS, Hkv, D, dtype=torch.bfloat16, device=DEV) * 0.5
    vcache = torch.randn(nblocks, BS, Hkv, D, dtype=torch.bfloat16, device=DEV) * 0.5
    bt = torch.zeros(pad_reqs, nblocks, dtype=torch.int32, device=DEV)
    bt[0] = torch.arange(nblocks, dtype=torch.int32, device=DEV)

    q = (torch.randn(qlen_real, Hq, D, dtype=torch.float32, device=DEV) * 0.5).to(torch.bfloat16)
    out = torch.zeros_like(q)

    att = SpecDecodeAttention(pad_reqs, Hq, D, torch.device(DEV), 10)
    att.run(q, kcache, vcache, out, cu, seqused, bt, scale, pad_reqs, qlen_real)

    ref = reference(q, kcache, vcache, bt, [kv_real], [qlen_real],
                    torch.tensor([0], dtype=torch.int32, device=DEV), Hq, Hkv, D, scale)
    err = (out.float() - ref).abs().max().item()
    print(f"stale-tail(G={Hq//Hkv},kv={kv_real}): max_abs_err={err:.4e} (no crash)")
    return err < 2e-2


ok = run_stale_tail_case() and ok
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

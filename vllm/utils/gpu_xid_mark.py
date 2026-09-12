"""Marker kernel used by the GPU fault-localization traces.

One store per launch: writes the encoded (layer, phase) into page-locked host
memory so the value survives a device fault. Deliberately trivial — it must not
perturb scheduling beyond a single kernel launch queue entry.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mark_kernel(mark_ptr, value, seq):
    tl.store(mark_ptr + 0, value)
    tl.store(mark_ptr + 1, seq)


def mark_kernel(mark_ptr: torch.Tensor, value: int, seq: int) -> None:
    _mark_kernel[(1,)](mark_ptr, value, seq)

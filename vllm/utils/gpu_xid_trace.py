"""Opt-in diagnostics for GPU illegal-address investigations.

Two independent knobs:

* ``GLM_XID_TRACE=1`` logs a one-line JSON record per instrumented stage.
  Shapes and dtypes only, so no device synchronization happens on this path.
* ``GLM_XID_SYNC_FILE=<path>`` enables a *single* device synchronization at
  whichever labelled boundary the file currently names. Bisecting that way
  keeps every other overlap in the step intact, so a sync point that hides the
  fault still localizes it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import torch


_ENABLED = os.getenv("GLM_XID_TRACE", "0") == "1"
_ABORT = os.getenv("GLM_XID_TRACE_ABORT", "0") == "1"
_SYNC_FILE = os.getenv("GLM_XID_SYNC_FILE", "")
_sync_point = ""
_sync_point_read_at = 0.0


def enabled() -> bool:
    return _ENABLED


def _capturing() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.is_initialized()
        and torch.cuda.is_current_stream_capturing()
    )


def sync_point(name: str) -> None:
    """Synchronize only when the live sync-point file currently names ``name``."""
    global _sync_point, _sync_point_read_at
    if not _SYNC_FILE or _capturing():
        return
    now = time.monotonic()
    if now - _sync_point_read_at > 0.5:
        try:
            with open(_SYNC_FILE) as handle:
                _sync_point = handle.read().strip()
        except OSError:
            _sync_point = ""
        _sync_point_read_at = now
    if _sync_point == name:
        torch.cuda.synchronize()


_DUMP_EVERY = 32
_EMIT_COUNT = 0


def emit(stage: str, **fields: Any) -> None:
    if not _ENABLED:
        return
    global _EMIT_COUNT
    _EMIT_COUNT += 1
    if _EMIT_COUNT % _DUMP_EVERY == 0:
        if _MARK_ENABLED:
            dump_marks()
        if os.getenv("GLM_XID_PROBE_FILE"):
            probe_dump(os.getenv("GLM_XID_PROBE_FILE"))
    record = {"stage": stage, **fields}
    print(
        "GLM_XID_TRACE " + json.dumps(record, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def tensor_summary(name: str, tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {name: None}
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
        }
    }


def check_bounds(
    stage: str,
    tensor: torch.Tensor | None,
    lower: int,
    upper: int,
    name: str,
) -> None:
    """Report bounds violations only when values are explicitly requested."""
    if not _ENABLED or tensor is None or tensor.numel() == 0 or _capturing():
        return
    if os.getenv("GLM_XID_TRACE_VALUES", "0") != "1":
        return
    invalid = ((tensor < lower) | (tensor >= upper)).any()
    if bool(invalid.item()):
        emit(
            stage,
            error="bounds",
            name=name,
            lower=lower,
            upper=upper,
            min=int(tensor.min().item()),
            max=int(tensor.max().item()),
        )
        if _ABORT:
            raise RuntimeError(f"{name} out of bounds at {stage}: [{lower}, {upper})")


def check_monotonic(stage: str, tensor: torch.Tensor | None, name: str) -> None:
    if not _ENABLED or tensor is None or tensor.numel() < 2 or _capturing():
        return
    if os.getenv("GLM_XID_TRACE_VALUES", "0") != "1":
        return
    if bool((tensor[1:] < tensor[:-1]).any().item()):
        emit(stage, error="non_monotonic", name=name)
        if _ABORT:
            raise RuntimeError(f"{name} is non-monotonic at {stage}")


# ---------------------------------------------------------------------------
# Stage markers.
#
# A device fault kills the context, taking every device buffer with it, so the
# only way to learn how far a step got is to have the GPU write progress into
# page-locked host memory as it goes. The marker kernel is a single store, it
# needs no host synchronization, and it works inside CUDA graph capture. After
# a fault the host reads ``_MARK_BUFFER``: ``before`` holds the last completed
# (step, layer, phase), so the fault is in the stage that follows it.
# ---------------------------------------------------------------------------

_MARK_BUFFER: list[int] = [0, 0, 0, 0]  # step, layer, phase, total_marks
_MARK_TENSOR: torch.Tensor | None = None
_MARK_ENABLED = os.getenv("GLM_XID_MARKS", "0") == "1"
_MARK_SEQ = 0


def _mark_tensor() -> torch.Tensor | None:
    global _MARK_TENSOR
    if not _MARK_ENABLED:
        return None
    if _MARK_TENSOR is None:
        try:
            _MARK_TENSOR = torch.zeros(4, dtype=torch.int32, pin_memory=True)
            _MARK_TENSOR.numpy()
        except Exception:
            return None
    return _MARK_TENSOR


def mark_prepare_accel() -> torch.Tensor | None:
    tensor = _mark_tensor()
    if tensor is None:
        return None
    try:
        import atexit

        atexit.register(dump_marks)
    except Exception:
        pass
    try:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        return get_accelerator_view_from_cpu_tensor(tensor)
    except Exception:
        return None


def mark(accel_view: torch.Tensor | None, layer: int, phase: int) -> None:
    """Enqueue the marker store for (layer, phase); never blocks the host.

    Runs during CUDA graph capture too: the store is recorded into the graph so
    replays keep reporting progress.
    """
    global _MARK_SEQ
    if accel_view is None:
        return
    if phase == 0:
        _MARK_SEQ += 1
    try:
        from vllm.utils.gpu_xid_mark import mark_kernel

        mark_kernel(accel_view, layer * 8 + phase, _MARK_SEQ)
    except Exception:
        pass


def dump_marks() -> None:
    tensor = _mark_tensor()
    if tensor is None:
        return
    values = tensor.numpy()
    _MARK_BUFFER[:] = [int(v) for v in values]
    sys.stderr.write(
        "GLM_XID_MARKS "
        + json.dumps(
            {"step": int(values[0]), "layer": int(values[1]), "phase": int(values[2])}
        )
        + "\n"
    )
    sys.stderr.flush()


def read_marks() -> dict[str, int]:
    tensor = _mark_tensor()
    if tensor is None:
        return {}
    values = tensor.numpy()
    return {
        "step": int(values[0]),
        "layer": int(values[1]),
        "phase": int(values[2]),
    }


SENTINEL = 0x7FFFFFFF
PROBE_CHECKS = 8
PROBE_CAPACITY = int(os.getenv("GLM_XID_PROBE_SLOTS", "4096"))
_probe_slots: torch.Tensor | None = None


def probe_slots() -> torch.Tensor | None:
    """Pinned host buffer holding one record per (check, request) slot.

    Layout: ``slots[check * PROBE_CAPACITY + req]``, pre-filled with -1. A
    kernel that detects an out-of-range index stores the offending value here
    and skips the access, so the evidence outlives the device fault.
    """
    global _probe_slots
    if _probe_slots is None:
        try:
            _probe_slots = torch.full(
                (PROBE_CHECKS * PROBE_CAPACITY,),
                -1,
                dtype=torch.int32,
                pin_memory=True,
            )
            _probe_slots.numpy()
        except Exception:
            _probe_slots = None
    return _probe_slots


GUARD_ENABLED = os.getenv("GLM_XID_COPY_GUARD", "0") == "1"


def probe_accel_view() -> torch.Tensor | None:
    # Off by default: the guard skips a copy it judges out of range, which can
    # turn a fault into silently stale state. It is a diagnostic, not the fix,
    # so the production path runs the plain copy.
    if not GUARD_ENABLED:
        return None
    slots = probe_slots()
    if slots is None:
        return None
    try:
        import atexit

        atexit.register(probe_dump, os.getenv("GLM_XID_PROBE_FILE") or None)
    except Exception:
        pass
    try:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        return get_accelerator_view_from_cpu_tensor(slots)
    except Exception:
        return None


_PROBE_CONTEXT = ""


def probe_context(**fields: Any) -> None:
    """Record launch-time context (block-table width, block size) in dumps."""
    global _PROBE_CONTEXT
    _PROBE_CONTEXT = " ".join(f"{k}={v}" for k, v in sorted(fields.items()))


def probe_report() -> str:
    """Human-readable dump of every recorded violation (host side, no CUDA)."""
    slots = probe_slots()
    if slots is None:
        return "probe buffer unavailable"
    values = slots.numpy()
    lines = []
    for check in range(PROBE_CHECKS):
        base = check * PROBE_CAPACITY
        hits = [(i, int(values[base + i])) for i in range(PROBE_CAPACITY)]
        hits = [(i, v) for i, v in hits if v != -1]
        if hits:
            lines.append(f"check={check} hits={len(hits)} {hits[:8]}")
    header = _PROBE_CONTEXT or "no context"
    body = "\n".join(lines) if lines else "no violations recorded"
    return header + "\n" + body


def probe_dump(path: str | None = None) -> None:
    text = probe_report()
    sys.stderr.write("GLM_XID_PROBE " + text.replace("\n", " | ") + "\n")
    sys.stderr.flush()
    if path:
        try:
            with open(path, "w") as handle:
                handle.write(text + "\n")
        except OSError:
            pass


_PROBE_TICK = 0


def probe_tick() -> None:
    """Dump probe records periodically; call sites sit on the hot path."""
    global _PROBE_TICK
    path = os.getenv("GLM_XID_PROBE_FILE")
    if not path:
        return
    _PROBE_TICK += 1
    if _PROBE_TICK % 32 == 0:
        probe_dump(path)


_VIOLATION_CAPACITY = int(os.getenv("GLM_XID_VIOLATION_SLOTS", "16384"))
_violation_slots: torch.Tensor | None = None
_violation_meta: torch.Tensor | None = None
_violation_calls = 0
_VIOLATION_DUMP = os.getenv("GLM_XID_VIOLATION_FILE", "")


def violation_buffers() -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Pinned host buffers a Triton kernel can write to and survive a fault.

    CUDA faults kill the device context, so any evidence that lives in device
    memory is lost at the moment it matters. These live in page-locked host
    memory, which the kernel reaches through its UVA mapping: writes land even
    if the same launch later trips an illegal access. Slots stay at
    ``SENTINEL`` unless a check fails, so post-mortem scanning finds the
    offending token indices and values without atomics or extra host syncs.
    """
    global _violation_slots, _violation_meta
    if _violation_slots is None:
        try:
            _violation_slots = torch.full(
                (_VIOLATION_CAPACITY,), SENTINEL, dtype=torch.int32, pin_memory=True
            )
            _violation_meta = torch.zeros(8, dtype=torch.int32, pin_memory=True)
            _violation_slots.numpy()  # force the host view to exist
            _violation_meta.numpy()
        except Exception:
            _violation_slots, _violation_meta = None, None
    return _violation_slots, _violation_meta


def violation_accel_views() -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Accelerator-side views of the violation buffers (same host memory)."""
    slots, meta = violation_buffers()
    if slots is None:
        return None, None
    try:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        return (
            get_accelerator_view_from_cpu_tensor(slots),
            get_accelerator_view_from_cpu_tensor(meta),
        )
    except Exception:
        return None, None


def note_violation_call(limit_a: int, limit_b: int) -> None:
    """Record call context in the pinned header and refresh the dump."""
    global _violation_calls
    slots, meta = violation_buffers()
    if meta is None:
        return
    _violation_calls += 1
    meta[0] = _violation_calls
    meta[1] = limit_a
    meta[2] = limit_b
    if _VIOLATION_DUMP and _violation_calls % 32 == 0:
        dump_violations()


def dump_violations() -> None:
    slots, meta = violation_buffers()
    if slots is None or not _VIOLATION_DUMP:
        return
    try:
        import numpy as np

        bad = np.nonzero(slots.numpy() != SENTINEL)[0]
        lines = [
            f"calls={int(meta.numpy()[0])} limit_a={int(meta.numpy()[1])} "
            f"limit_b={int(meta.numpy()[2])} violations={len(bad)}"
        ]
        lines += [f"  slot={int(i)} value={int(slots[int(i)])}" for i in bad[:64]]
        with open(_VIOLATION_DUMP, "w") as handle:
            handle.write("\n".join(lines) + "\n")
    except Exception:
        pass


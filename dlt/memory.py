"""Per-rank memory accounting measured from the live tensors a rank owns.

This works identically on CPU and GPU, so the sharding claims (FSDP stores
~1/N of params, grads and optimizer state per rank) are *measured* even without
GPUs. On CUDA we additionally record the allocator's peak, which also includes
activations, all-gather buffers, and fragmentation.
"""
from __future__ import annotations

import torch


def _local(t: torch.Tensor) -> torch.Tensor:
    # FSDP2 parameters/grads are DTensors; to_local() is this rank's shard.
    return t.to_local() if hasattr(t, "to_local") else t


def _nbytes(t: torch.Tensor) -> int:
    t = _local(t)
    return t.numel() * t.element_size()


def state_bytes(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    params = {id(p): p for p in model.parameters()}.values()
    p_bytes = sum(_nbytes(p) for p in params)
    g_bytes = sum(_nbytes(p.grad) for p in params if p.grad is not None)
    o_bytes = 0
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v) and v.dim() > 0:
                o_bytes += _nbytes(v)
    out = {"param_bytes": p_bytes, "grad_bytes": g_bytes, "optim_bytes": o_bytes,
           "state_bytes": p_bytes + g_bytes + o_bytes}
    if torch.cuda.is_available():
        out["cuda_peak_allocated"] = torch.cuda.max_memory_allocated()
        out["cuda_peak_reserved"] = torch.cuda.max_memory_reserved()
    return out


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

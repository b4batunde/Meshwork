"""Process-group setup and the three parallelism strategies under study:

    single : plain model on one device (the baseline)
    ddp    : DistributedDataParallel. Every GPU holds a full replica of
             params, grads and optimizer state; grads are all-reduced in
             buckets that overlap with the backward pass.
    fsdp   : Fully Sharded Data Parallel (FSDP2 / `fully_shard`). Params,
             grads and optimizer state are sharded across GPUs; each block's
             params are all-gathered just before use and grads are
             reduce-scattered after backward.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1 or dist.is_initialized()


def init_distributed(force_cpu: bool = False) -> DistInfo:
    use_cuda = torch.cuda.is_available() and not force_cpu
    if "RANK" not in os.environ:  # launched with plain `python`
        dev = torch.device("cuda:0" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.set_device(dev)
        return DistInfo(device=dev)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if use_cuda:
        dev = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(dev)
        dist.init_process_group("nccl", device_id=dev)
    else:
        dev = torch.device("cpu")
        dist.init_process_group("gloo")
    return DistInfo(rank, local_rank, world_size, dev)


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def wrap_model(model: nn.Module, strategy: str, info: DistInfo, *, half: torch.dtype | None,
               bucket_cap_mb: int = 25, reshard_after_forward: bool = True,
               ddp_bf16_comm: bool = False) -> nn.Module:
    if strategy == "single":
        if info.world_size > 1:
            raise ValueError("strategy=single requires exactly one process")
        return model.to(info.device)

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = model.to(info.device)
        ddp = DDP(
            model,
            device_ids=[info.local_rank] if info.device.type == "cuda" else None,
            bucket_cap_mb=bucket_cap_mb,
            gradient_as_bucket_view=True,
        )
        if ddp_bf16_comm:
            # Halve all-reduce bytes: gradients are cast to 16-bit for communication only.
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            hook = (default_hooks.fp16_compress_hook if half == torch.float16
                    else default_hooks.bf16_compress_hook)
            ddp.register_comm_hook(None, hook)
        return ddp

    if strategy == "fsdp":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mesh = init_device_mesh(info.device.type, (info.world_size,))
        mp = MixedPrecisionPolicy(param_dtype=half, reduce_dtype=torch.float32) if half else MixedPrecisionPolicy()
        meta = any(p.is_meta for p in model.parameters())
        if not meta:
            model = model.to(info.device)
        # Shard each transformer block as its own unit so only one block's full
        # params are materialized at a time; the root picks up embeddings/head.
        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
        fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
        if meta:
            # Built on the meta device: allocate only this rank's shards, then initialize them.
            # This is what lets FSDP construct models larger than one GPU (or host) can hold.
            model.to_empty(device=info.device)
            model.init_weights()
            assert model.lm_head.weight is model.wte.weight, "weight tying lost during meta init"
        return model

    raise ValueError(f"unknown strategy {strategy!r}")


def grad_sync_context(model: nn.Module, strategy: str, sync: bool):
    """Skip gradient communication on non-final gradient-accumulation steps
    (or everywhere, when measuring compute-only time)."""
    if strategy == "ddp" and not sync:
        return model.no_sync()
    if strategy == "fsdp":
        model.set_requires_gradient_sync(sync)
    return contextlib.nullcontext()


def all_reduce_mean(x: float, info: DistInfo) -> float:
    if not dist.is_initialized():
        return x
    t = torch.tensor([x], dtype=torch.float64, device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / info.world_size).item()


def all_reduce_max(x: float, info: DistInfo) -> float:
    if not dist.is_initialized():
        return x
    t = torch.tensor([x], dtype=torch.float64, device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()

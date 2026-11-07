"""Single-device / DDP / FSDP trainer for the GPT model.

Launch:
  python -m dlt.train --strategy single --model gpt-125m
  torchrun --nproc_per_node 4 -m dlt.train --strategy ddp  --model gpt-125m
  torchrun --nproc_per_node 4 -m dlt.train --strategy fsdp --model gpt-350m

Runs on CUDA (NCCL) when available, otherwise on CPU (gloo) so the full
distributed code path can be validated on a laptop. Rank 0 writes a JSON
record with throughput, per-rank memory, loss curve, and (optionally) a
"no-sync" probe that measures how much step time is exposed communication.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import time

import torch
import torch.distributed as dist

from dlt.data import TokenSource
from dlt.memory import state_bytes
from dlt.model import CONFIGS, GPT


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--model", default="tiny", choices=list(CONFIGS))
    p.add_argument("--seq-len", type=int, default=None, help="override config seq_len")
    p.add_argument("--micro-batch", type=int, default=8, help="sequences per rank per micro-step")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=3, help="excluded from timing")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default=None,
                   help="compute dtype (default: bf16 on CUDA, fp32 on CPU)")
    p.add_argument("--act-ckpt", action="store_true", help="activation checkpointing per block")
    p.add_argument("--data", default="synthetic", help="synthetic | stdlib | path/to/tokens.bin")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--bucket-mb", type=float, default=25.0, help="DDP gradient bucket size")
    p.add_argument("--no-reshard-after-forward", action="store_true",
                   help="FSDP: keep params gathered between fwd and bwd (ZeRO-2 style)")
    p.add_argument("--meta-init", action="store_true",
                   help="FSDP: build on meta device and materialize only local shards (big models)")
    p.add_argument("--comm-probe-steps", type=int, default=0,
                   help="extra steps with gradient sync disabled to measure exposed comm time")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads per rank")
    p.add_argument("--out", default=None, help="JSON output path (rank 0)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def setup_dist(strategy):
    if strategy == "single":
        assert int(os.environ.get("WORLD_SIZE", 1)) == 1, "use ddp/fsdp with torchrun"
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, dev
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device("cpu")
    dist.init_process_group(backend=backend, device_id=dev if dev.type == "cuda" else None)
    return rank, world, dev


def build(args, rank, world, dev):
    cfg = CONFIGS[args.model]
    if args.seq_len:
        cfg.seq_len = args.seq_len
    torch.manual_seed(args.seed)  # identical init on every rank
    meta = args.meta_init and args.strategy == "fsdp"
    with torch.device("meta") if meta else contextlib.nullcontext():
        model = GPT(cfg)
    model.activation_checkpointing = args.act_ckpt
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.strategy == "fsdp":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh(dev.type, (world,))
        mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
        reshard = not args.no_reshard_after_forward
        for blk in model.blocks:  # one FSDP unit per transformer block
            fully_shard(blk, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard)
        fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard)
        if meta:
            model.to_empty(device=dev)
            model.init_weights()
        wrapped = model
    else:
        model.to(dev)
        wrapped = model
        if args.strategy == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP
            wrapped = DDP(model, device_ids=[dev.index] if dev.type == "cuda" else None,
                          bucket_cap_mb=args.bucket_mb, gradient_as_bucket_view=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        fused=(dev.type == "cuda"))  # fused also works on FSDP2 DTensors (measured: 89 -> 48 ms/step)
    return cfg, model, wrapped, opt, dtype


def lr_at(step, args):
    warm = max(1, args.steps // 10)
    if step < warm:
        return args.lr * (step + 1) / warm
    prog = (step - warm) / max(1, args.steps - warm)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def train_step(step, args, data, wrapped, model, opt, rank, world, dev, dtype, grad_sync=True):
    for g in opt.param_groups:
        g["lr"] = lr_at(step, args)
    autocast = (torch.autocast(dev.type, dtype=dtype)
                if (dtype != torch.float32 and args.strategy != "fsdp") else contextlib.nullcontext())
    total = torch.zeros((), device=dev)
    for micro in range(args.grad_accum):
        x, y = data.local_batch(step * args.grad_accum + micro, args.micro_batch, rank, world)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        last = micro == args.grad_accum - 1
        do_sync = grad_sync and last
        if args.strategy == "ddp" and not do_sync:
            ctx = wrapped.no_sync()
        else:
            ctx = contextlib.nullcontext()
        if args.strategy == "fsdp":
            model.set_requires_gradient_sync(do_sync)
        with ctx, autocast:
            _, loss = wrapped(x, y)
            (loss / args.grad_accum).backward()
        total += loss.detach() / args.grad_accum
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=False)  # keep grads allocated so memory is measurable
    return total


def main(argv=None):
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    rank, world, dev = setup_dist(args.strategy)
    if args.dtype is None:
        args.dtype = "bf16" if dev.type == "cuda" else "fp32"
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log = (lambda *a: print(*a, flush=True)) if (rank == 0 and not args.quiet) else (lambda *a: None)

    cfg, model, wrapped, opt, dtype = build(args, rank, world, dev)
    data = TokenSource(args.data, cfg.vocab_size, cfg.seq_len, seed=args.seed)
    n_params = cfg.num_params()
    tokens_per_step = args.micro_batch * args.grad_accum * world * cfg.seq_len
    log(f"[{args.strategy}] model={args.model} params={n_params/1e6:.1f}M world={world} "
        f"device={dev.type} dtype={args.dtype} tokens/step={tokens_per_step}")

    status, losses, step_times = "ok", [], []
    try:
        for step in range(args.steps):
            sync(dev)
            t0 = time.perf_counter()
            loss = train_step(step, args, data, wrapped, model, opt, rank, world, dev, dtype)
            if world > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            sync(dev)
            dt = time.perf_counter() - t0
            losses.append(loss.item())
            if step >= args.warmup_steps:
                step_times.append(dt)
            if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
                log(f"  step {step:4d} loss {losses[-1]:.4f} {dt*1000:8.1f} ms "
                    f"{tokens_per_step/dt:10.0f} tok/s")
        mem = state_bytes(model, opt)

        probe_times = []
        if args.comm_probe_steps > 0 and world > 1:
            for i in range(args.comm_probe_steps):
                sync(dev)
                t0 = time.perf_counter()
                train_step(args.steps + i, args, data, wrapped, model, opt, rank, world, dev, dtype,
                           grad_sync=False)
                sync(dev)
                if i > 0:
                    probe_times.append(time.perf_counter() - t0)
    except torch.OutOfMemoryError:
        status, mem, probe_times = "oom", {}, []
        log("  CUDA OOM")

    per_rank_mem = [None] * world
    if world > 1:
        dist.all_gather_object(per_rank_mem, mem)
    else:
        per_rank_mem = [mem]

    if rank == 0:
        def median(v):
            v = sorted(v)
            return v[len(v) // 2] if v else None
        st = median(step_times)
        rec = {
            "label": "Measured",
            "strategy": args.strategy, "model": args.model, "world_size": world,
            "params": n_params, "config": cfg.to_dict(), "device": dev.type,
            "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor() or "cpu",
            "dtype": args.dtype, "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
            "act_ckpt": args.act_ckpt, "tokens_per_step": tokens_per_step,
            "status": status,
            "step_time_s": st,
            "tokens_per_s": tokens_per_step / st if st else None,
            "nosync_step_time_s": median(probe_times),
            "losses": losses,
            "per_rank_memory": per_rank_mem,
            "torch": torch.__version__,
        }
        if rec["nosync_step_time_s"] and st:
            rec["exposed_comm_s"] = max(0.0, st - rec["nosync_step_time_s"])
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(rec, f, indent=2)
        log(f"  median step {st*1000 if st else float('nan'):.1f} ms | "
            f"state/rank {mem.get('state_bytes', 0)/2**20:.1f} MB | status={status}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

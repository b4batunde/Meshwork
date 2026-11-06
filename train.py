"""Train a GPT-style model with single-GPU, DDP or FSDP, and record
throughput / memory / communication metrics for the scaling study.

Examples
--------
  # baseline
  python train.py --model gpt-125m --strategy single --micro-batch 16 --steps 50
  # 4-GPU DDP and FSDP
  torchrun --nproc_per_node 4 train.py --model gpt-125m --strategy ddp  --micro-batch 16 --steps 50
  torchrun --nproc_per_node 4 train.py --model gpt-125m --strategy fsdp --micro-batch 16 --steps 50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import torch

from dlt.data import TokenBatcher
from dlt.distributed import (all_reduce_max, all_reduce_mean, cleanup, grad_sync_context,
                             init_distributed, wrap_model)
from dlt.model import GPT, flops_per_token, get_config, num_params

# Dense bf16 peak TFLOPs, used for MFU. Override with --peak-tflops.
PEAK_TFLOPS = {"A100": 312.0, "H100": 989.0, "H200": 989.0, "L40S": 362.0,
               "A6000": 155.0, "A10": 125.0, "T4": 65.0, "V100": 125.0, "4090": 165.0, "3090": 71.0}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt-125m")
    p.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--micro-batch", type=int, default=8, help="sequences per GPU per micro-step")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=None, help="defaults to model block_size")
    p.add_argument("--steps", type=int, default=50, help="optimizer steps")
    p.add_argument("--timing-warmup", type=int, default=5, help="steps excluded from timing")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--lr-warmup", type=int, default=None, help="default: 10%% of steps")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--data", default=None, help="uint16 token .bin file; omit for synthetic tokens")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="fp16 (with loss scaling) for GPUs without bf16, e.g. T4/V100")
    p.add_argument("--compile", action="store_true", help="torch.compile each block")
    p.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing")
    p.add_argument("--bucket-cap-mb", type=int, default=25, help="DDP all-reduce bucket size")
    p.add_argument("--ddp-bf16-comm", action="store_true",
                   help="DDP: all-reduce gradients in bf16 (bf16_compress_hook), halving comm bytes")
    p.add_argument("--meta-init", action="store_true",
                   help="FSDP: build the model on the meta device and materialize only local shards")
    p.add_argument("--no-reshard", action="store_true",
                   help="FSDP: keep params gathered after forward (ZeRO-2-like)")
    p.add_argument("--no-grad-sync", action="store_true",
                   help="disable gradient communication (NOT valid training; used to "
                        "measure compute-only step time and isolate comm overhead)")
    p.add_argument("--profile", action="store_true", help="torch.profiler pass: comm vs compute kernel time")
    p.add_argument("--gpu-type", default=None, help="for MFU, e.g. A100, H100")
    p.add_argument("--peak-tflops", type=float, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", default="results/runs.jsonl")
    p.add_argument("--tag", default="", help="free-form label stored with the result")
    p.add_argument("--cpu", action="store_true", help="force CPU/gloo (for tests)")
    return p.parse_args(argv)


def detect_gpu_type() -> str | None:
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    for key in PEAK_TFLOPS:
        if key in name:
            return key
    return name


def lr_at(step: int, args) -> float:
    warm = args.lr_warmup if args.lr_warmup is not None else max(1, args.steps // 10)
    if step < warm:
        return args.lr * (step + 1) / warm
    progress = (step - warm) / max(1, args.steps - warm)
    min_lr = args.lr * args.min_lr_ratio
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def local_numel(t: torch.Tensor) -> int:
    return t.to_local().numel() if hasattr(t, "to_local") else t.numel()


def local_bytes(t: torch.Tensor) -> int:
    return local_numel(t) * t.element_size()


def model_state_bytes(model, optimizer) -> dict:
    """Bytes of params / grads / optimizer state that live on *this* rank.
    Works on CPU too, so memory savings can be validated without GPUs."""
    params = sum(local_bytes(p) for p in model.parameters())
    grads = sum(local_bytes(p.grad) for p in model.parameters() if p.grad is not None)
    optim = 0
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v) and v.dim() > 0:
                optim += local_bytes(v)
    return {"params": params, "grads": grads, "optimizer": optim, "total": params + grads + optim}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv=None):
    args = parse_args(argv)
    info = init_distributed(force_cpu=args.cpu)
    dev = info.device
    on_gpu = dev.type == "cuda"
    half = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.dtype) if on_gpu else None

    torch.manual_seed(args.seed)  # identical init on every rank
    if on_gpu:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = get_config(args.model, grad_ckpt=args.grad_ckpt or None)
    seq_len = args.seq_len or cfg.block_size
    n_params = num_params(cfg)

    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": args.tag, "model": args.model, "n_params": n_params,
        "strategy": args.strategy, "world_size": info.world_size,
        "micro_batch": args.micro_batch, "grad_accum": args.grad_accum, "seq_len": seq_len,
        "global_batch_tokens": args.micro_batch * args.grad_accum * seq_len * info.world_size,
        "dtype": args.dtype if half else "fp32", "compile": args.compile, "grad_ckpt": args.grad_ckpt,
        "no_grad_sync": args.no_grad_sync, "reshard_after_forward": not args.no_reshard,
        "bucket_cap_mb": args.bucket_cap_mb, "ddp_bf16_comm": args.ddp_bf16_comm, "synthetic_data": args.data is None,
        "device": torch.cuda.get_device_name(0) if on_gpu else "cpu",
    }

    status = "ok"
    try:
        if args.meta_init and args.strategy == "fsdp":
            with torch.device("meta"):
                model = GPT(cfg)
        else:
            model = GPT(cfg)
        if args.compile:
            for block in model.blocks:
                block.compile()
        model = wrap_model(model, args.strategy, info, half=half,
                           bucket_cap_mb=args.bucket_cap_mb,
                           reshard_after_forward=not args.no_reshard,
                           ddp_bf16_comm=args.ddp_bf16_comm)

        decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        )

        batcher = TokenBatcher(args.data, seq_len, args.micro_batch, info.rank, info.world_size,
                               cfg.vocab_size, seed=args.seed, device=dev)

        # DDP (and single) use autocast; FSDP does its own cast via MixedPrecisionPolicy.
        use_autocast = half is not None and args.strategy != "fsdp"
        # fp16 needs loss scaling to avoid gradient underflow; bf16 does not.
        scaler = torch.amp.GradScaler(dev.type, enabled=half == torch.float16)

        state_mem = {}

        def train_step(step: int) -> float:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, args)
            total = 0.0
            for micro in range(args.grad_accum):
                last = micro == args.grad_accum - 1
                x, y = batcher.get(step, micro)
                with grad_sync_context(model, args.strategy, sync=last and not args.no_grad_sync):
                    with torch.autocast(dev.type, dtype=half or torch.bfloat16, enabled=use_autocast):
                        _, loss = model(x, y)
                    scaler.scale(loss / args.grad_accum).backward()
                total += loss.detach()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if not state_mem:  # measure once, while grads and optimizer state both exist
                state_mem.update(model_state_bytes(model, optimizer))
            optimizer.zero_grad(set_to_none=True)
            return (total / args.grad_accum).item()

        if on_gpu:
            torch.cuda.reset_peak_memory_stats(dev)

        losses, step_times = [], []
        for step in range(args.steps):
            sync(dev)
            t0 = time.perf_counter()
            loss = train_step(step)
            sync(dev)
            dt = time.perf_counter() - t0
            if step >= args.timing_warmup:
                step_times.append(dt)
            loss = all_reduce_mean(loss, info)
            losses.append(loss)
            if info.is_main and (step % args.log_every == 0 or step == args.steps - 1):
                tps = record["global_batch_tokens"] / dt
                print(f"step {step:5d} | loss {loss:.4f} | {dt*1000:7.1f} ms | {tps:,.0f} tok/s", flush=True)

        comm = run_profile(train_step, args, dev) if (args.profile and on_gpu) else None

        # Use the slowest rank's median step time: that's what everyone waits for.
        med = statistics.median(step_times) if step_times else float("nan")
        med = all_reduce_max(med, info)
        tok_s = record["global_batch_tokens"] / med
        gpu_type = args.gpu_type or detect_gpu_type()
        peak = args.peak_tflops or PEAK_TFLOPS.get(gpu_type or "", None)
        achieved_tflops = flops_per_token(cfg, seq_len) * tok_s / info.world_size / 1e12
        record.update({
            "step_time_median_s": med,
            "tokens_per_s": tok_s,
            "tokens_per_s_per_gpu": tok_s / info.world_size,
            "tflops_per_gpu": achieved_tflops,
            "mfu": achieved_tflops / peak if peak else None,
            "gpu_type": gpu_type,
            "final_loss": losses[-1],
            "losses": losses,
            "state_bytes_per_rank": {k: all_reduce_max(v, info) for k, v in state_mem.items()},
            "comm_profile": comm,
        })
        if on_gpu:
            record["peak_mem_allocated_gb"] = all_reduce_max(torch.cuda.max_memory_allocated(dev) / 2**30, info)
            record["peak_mem_reserved_gb"] = all_reduce_max(torch.cuda.max_memory_reserved(dev) / 2**30, info)
            record["gpu_mem_total_gb"] = torch.cuda.get_device_properties(dev).total_memory / 2**30

    except torch.OutOfMemoryError:
        status = "oom"
        if info.is_main:
            print("OUT OF MEMORY", file=sys.stderr, flush=True)

    record["status"] = status
    if info.is_main:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(record) + "\n")
        if status == "ok":
            mem = record.get("peak_mem_allocated_gb")
            mfu = f"{record['mfu']:.1%}" if record["mfu"] is not None else "n/a"
            mem = f"{mem:.2f} GiB" if mem is not None else "n/a"
            print(f"[{args.strategy} x{info.world_size}] {record['tokens_per_s']:,.0f} tok/s | MFU {mfu} | "
                  f"peak mem {mem} | model state/rank "
                  f"{record['state_bytes_per_rank']['total']/2**20:,.1f} MiB", flush=True)

    if status == "oom":
        # Ranks may be out of step after an OOM, so skip the barrier.
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        sys.exit(3)
    cleanup()


def run_profile(train_step, args, dev, steps: int = 3) -> dict:
    """Split GPU kernel time into NCCL communication vs everything else.
    Communication that overlaps compute still shows up here, so pair this with
    a --no-grad-sync run to get the *exposed* (non-overlapped) cost."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for s in range(steps):
            train_step(args.steps + s)
        sync(dev)
    comm_us = compute_us = 0.0
    for evt in prof.key_averages():
        t = getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)
        if "nccl" in evt.key.lower():
            comm_us += t
        else:
            compute_us += t
    total = comm_us + compute_us
    return {"comm_ms_per_step": comm_us / 1e3 / steps, "compute_ms_per_step": compute_us / 1e3 / steps,
            "comm_frac_of_kernel_time": comm_us / total if total else None}


if __name__ == "__main__":
    main()

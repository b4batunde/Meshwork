"""Collective microbenchmark (like nccl-tests, but tiny).

  torchrun --nproc_per_node 4 -m dlt.commbench --out results/gpu/commbench_w4.json

Times all_reduce, all_gather and reduce_scatter over message sizes and reports
algorithm bandwidth and *bus* bandwidth (nccl-tests convention), plus a
latency/bandwidth fit  t(n) = alpha + factor(op, p) * n / busbw.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

# bytes each rank must move per byte of the full buffer, for ring algorithms
def bus_factor(op: str, p: int) -> float:
    return 2 * (p - 1) / p if op == "all_reduce" else (p - 1) / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pow", type=int, default=12)
    ap.add_argument("--max-pow", type=int, default=26)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, p = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(dev)
    dtype = torch.bfloat16 if cuda else torch.float32

    rows = []
    for pw in range(args.min_pow, args.max_pow + 1):
        nbytes = 2 ** pw
        n = nbytes // torch.tensor([], dtype=dtype).element_size()
        n -= n % p
        full = torch.randn(n, device=dev, dtype=dtype)
        shard = torch.empty(n // p, device=dev, dtype=dtype)
        ops = {
            "all_reduce": lambda: dist.all_reduce(full),
            "all_gather": lambda: dist.all_gather_into_tensor(full, shard),
            "reduce_scatter": lambda: dist.reduce_scatter_tensor(shard, full),
        }
        for op, fn in ops.items():
            try:
                fn()  # warmup
            except (RuntimeError, NotImplementedError):
                continue
            ts = []
            for _ in range(args.iters):
                dist.barrier()
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                if cuda:
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            t = torch.tensor(sorted(ts)[len(ts) // 2], device=dev, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            t = t.item()
            size = n * full.element_size()
            rows.append({"op": op, "bytes": size, "time_s": t,
                         "algbw_GBps": size / t / 1e9,
                         "busbw_GBps": size * bus_factor(op, p) / t / 1e9})

    # least-squares fit per op: t = alpha + (factor * bytes) * inv_bw
    fits = {}
    for op in {r["op"] for r in rows}:
        xs = [r["bytes"] * bus_factor(op, p) for r in rows if r["op"] == op]
        ys = [r["time_s"] for r in rows if r["op"] == op]
        X = torch.tensor([[1.0, x] for x in xs], dtype=torch.float64)
        Y = torch.tensor(ys, dtype=torch.float64)
        sol = torch.linalg.lstsq(X, Y).solution
        alpha, inv_bw = max(sol[0].item(), 0.0), max(sol[1].item(), 1e-15)
        fits[op] = {"alpha_s": alpha, "busbw_GBps": 1 / inv_bw / 1e9}

    if rank == 0:
        rec = {"label": "Measured", "world_size": p, "backend": dist.get_backend(),
               "device": torch.cuda.get_device_name() if cuda else "cpu",
               "dtype": str(dtype), "rows": rows, "fits": fits}
        for op, f in sorted(fits.items()):
            print(f"{op:15s} alpha={f['alpha_s']*1e6:8.1f} us  busbw={f['busbw_GBps']:8.2f} GB/s")
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(rec, fh, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

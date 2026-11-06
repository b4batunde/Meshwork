"""Find the largest model each strategy can train on N GPUs (micro-batch 1).

Walks model sizes from small to large and stops a strategy at its first OOM.
The ratio (largest FSDP params / largest DDP params) is the "Z x larger
models" number in the headline.

    python bench/find_max_model.py --nproc 4
    python bench/find_max_model.py --nproc 4 --grad-ckpt   # also try with activation checkpointing
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlt.model import MODEL_CONFIGS, get_config, num_params  # noqa: E402

SIZES = [m for m in MODEL_CONFIGS if m != "tiny"]


def try_run(model, strategy, nproc, args) -> str:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           str(ROOT / "train.py"), "--model", model, "--strategy", strategy, "--micro-batch", "1",
           "--steps", "4", "--timing-warmup", "1", "--out", args.out, "--tag", "maxsize"]
    if args.grad_ckpt:
        cmd.append("--grad-ckpt")
    if strategy == "fsdp":
        cmd.append("--meta-init")  # never materialize the full model on one device
    if args.seq_len:
        cmd += ["--seq-len", str(args.seq_len)]
    print(f">>> {strategy:4s} x{nproc}  {model:9s} ({num_params(get_config(model))/1e6:,.0f}M params) ... ",
          end="", flush=True)
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print("timeout (treated as does-not-fit)")
        return "timeout"
    if r.returncode == 0:
        print("fits")
        return "ok"
    oom = r.returncode == 3 or "out of memory" in (r.stderr + r.stdout).lower()
    print("OOM" if oom else f"failed (exit {r.returncode})")
    if not oom:
        print(r.stderr[-2000:])
    return "oom" if oom else "error"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--strategies", nargs="+", default=["ddp", "fsdp"])
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--out", default="results/maxsize.jsonl")
    args = p.parse_args()

    largest = {}
    for strat in args.strategies:
        largest[strat] = None
        for model in SIZES:
            if try_run(model, strat, args.nproc, args) != "ok":
                break
            largest[strat] = model

    summary = {"nproc": args.nproc, "grad_ckpt": args.grad_ckpt,
               "largest": {s: (m, num_params(get_config(m)) if m else 0) for s, m in largest.items()}}
    ddp, fsdp = summary["largest"].get("ddp"), summary["largest"].get("fsdp")
    if ddp and fsdp and ddp[1]:
        summary["fsdp_over_ddp"] = fsdp[1] / ddp[1]
        print(f"\nLargest DDP: {ddp[0]}  |  largest FSDP: {fsdp[0]}  ->  {summary['fsdp_over_ddp']:.1f}x larger")
    with open(ROOT / "results" / "maxsize_summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()

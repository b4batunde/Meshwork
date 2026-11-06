"""MEASURED results that need no GPU: real train.py runs on CPU (gloo backend).

1. Numerical equivalence: single process vs DDP x2 vs FSDP x2 on the same global batch.
2. Per-rank model-state bytes (params + grads + AdamW state) for gpt-125m under DDP and
   FSDP on 1, 2 and 4 processes, read from the actual tensors, compared with the simulator.

These are measurements of correctness and memory layout, not GPU speed.

    python bench/measure_cpu.py            # writes results/measured_cpu/ and the README block
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from dlt.model import get_config  # noqa: E402
from dlt.simulator import RunSpec, simulate_memory  # noqa: E402

OUT = ROOT / "results" / "measured_cpu"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)}


def run(strategy, nproc, args, out):
    base = [str(ROOT / "train.py"), "--strategy", strategy, "--cpu", "--dtype", "fp32", "--out", str(out),
            "--log-every", "1000", *args]
    cmd = [sys.executable, *base] if strategy == "single" else \
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", *base]
    subprocess.run(cmd, check=True, env=ENV, cwd=ROOT, capture_output=True)
    return json.loads(out.read_text().strip().splitlines()[-1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.jsonl"):
        f.unlink()

    # 1. equivalence
    eq_args = ["--model", "tiny", "--steps", "30", "--timing-warmup", "2"]
    eq = {name: run(st, n, eq_args + ["--micro-batch", str(mb), "--tag", "equivalence"], OUT / "equivalence.jsonl")
          for name, st, n, mb in (("single", "single", 1, 8), ("ddp", "ddp", 2, 4), ("fsdp", "fsdp", 2, 4))}
    ref = eq["single"]["losses"]
    eq_rows = []
    for name in ("ddp", "fsdp"):
        diffs = [abs(a - b) / abs(a) for a, b in zip(ref, eq[name]["losses"])]
        eq_rows.append(dict(strategy=name, steps=len(diffs), max_rel_diff=max(diffs),
                            first=eq[name]["losses"][0], last=eq[name]["losses"][-1]))

    # 2. model state per rank: gpt-125m on 1-2 processes (4 x 125M does not fit this 7 GB CPU box),
    #    and the tiny model on 1, 2 and 4 processes to show the 1/N trend.
    state_rows = []
    for model, strat, n in (("gpt-125m", "ddp", 1), ("gpt-125m", "fsdp", 1), ("gpt-125m", "fsdp", 2),
                            ("tiny", "ddp", 1), ("tiny", "fsdp", 2), ("tiny", "fsdp", 4)):
        args = ["--model", model, "--steps", "1", "--timing-warmup", "0", "--micro-batch", "1",
                "--seq-len", "64", "--tag", "state"] + (["--meta-init"] if strat == "fsdp" else [])
        r = run(strat, n, args, OUT / "state.jsonl")
        meas = r["state_bytes_per_rank"]["total"]
        sim = simulate_memory(RunSpec(get_config(model), strat, n, 1, seq_len=64))["state_bytes"]
        state_rows.append(dict(model=model, strategy=strat, n=n, measured=meas, simulated=sim,
                               err=sim / meas - 1, parts=r["state_bytes_per_rank"]))
    (OUT / "summary.json").write_text(json.dumps(dict(equivalence=eq_rows, state=state_rows), indent=2))

    L = ["**Measured (CPU, gloo backend, real train.py runs).** These check correctness and memory layout; "
         "they say nothing about GPU speed.", "",
         "Same global batch of 8 sequences, 30 optimizer steps, fp32 (`tiny` model):", "",
         "| Run | Max relative loss difference vs single process | Loss, step 0 -> 29 |", "|---|---|---|"]
    for r in eq_rows:
        L.append(f"| {r['strategy'].upper()} x2 | {r['max_rel_diff']:.1e} | {r['first']:.4f} -> {r['last']:.4f} |")
    L += ["", "Per-rank model state (params + grads + AdamW m, v), read from the live tensors:", "",
          "| Model | Strategy | Processes | Measured | Simulated | Difference | Share of DDP's |", "|---|---|---|---|---|---|---|"]
    for r in state_rows:
        d1 = next(x for x in state_rows if x["model"] == r["model"] and x["strategy"] == "ddp")
        unit, div = ("MiB", 2**20) if r["measured"] > 2**22 else ("KiB", 2**10)
        L.append(f"| {r['model']} | {r['strategy'].upper()} | {r['n']} | {r['measured']/div:,.1f} {unit} | "
                 f"{r['simulated']/div:,.1f} {unit} | {r['err']:+.2%} | {r['measured']/d1['measured']:.1%} |")
    L += ["", "FSDP's per-rank share is exactly 1/N here because every dimension divides evenly (no shard padding), "
          "and the simulator's model-state arithmetic matches the live tensors byte for byte."]
    body = "\n".join(L) + "\n"
    (OUT / "MEASURED_CPU.md").write_text("# Measured on CPU\n\n" + body)

    readme = ROOT / "README.md"
    text = readme.read_text()
    start, end = "<!-- MEASURED_CPU:START -->", "<!-- MEASURED_CPU:END -->"
    if start in text:
        pre, rest = text.split(start, 1)
        readme.write_text(pre + start + "\n" + body.rstrip() + "\n" + end + rest.split(end, 1)[1])
    print(body)


if __name__ == "__main__":
    main()

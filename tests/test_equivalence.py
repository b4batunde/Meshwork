"""Correctness checks that run on CPU (gloo), no GPUs required.

1. Same global batch => single-process, DDP and FSDP must produce the same
   loss curve (up to float reduction-order noise). This proves the parallel
   versions are the *same training run*, just distributed.
2. FSDP must hold ~1/N of the model state (params + grads + Adam) per rank.

Run:  pytest -q tests/
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(ROOT)}
COMMON = ["--model", "tiny", "--steps", "15", "--cpu", "--dtype", "fp32", "--log-every", "100",
          "--timing-warmup", "2"]


def run(tmp_path, strategy, nproc, micro_batch, extra=()):
    out = tmp_path / f"{strategy}{nproc}.jsonl"
    args = [str(ROOT / "train.py"), "--strategy", strategy, "--micro-batch", str(micro_batch),
            "--out", str(out), *COMMON, *extra]
    if nproc == 1:
        cmd = [sys.executable, *args]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={nproc}", *args]
    subprocess.run(cmd, check=True, env=ENV, cwd=ROOT, capture_output=True)
    return json.loads(out.read_text().strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    return {
        "single": run(tmp, "single", 1, 8),
        "ddp": run(tmp, "ddp", 2, 4),
        "fsdp": run(tmp, "fsdp", 2, 4),
        "ddp_accum": run(tmp, "ddp", 2, 2, ["--grad-accum", "2"]),
        "fsdp_accum": run(tmp, "fsdp", 2, 2, ["--grad-accum", "2"]),
        "fsdp_meta": run(tmp, "fsdp", 2, 4, ["--meta-init"]),
    }


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_matches_single_process(runs, strategy):
    ref, got = runs["single"]["losses"], runs[strategy]["losses"]
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert b == pytest.approx(a, rel=1e-4), (ref, got)


def test_grad_accum_ddp_matches_fsdp(runs):
    for a, b in zip(runs["ddp_accum"]["losses"], runs["fsdp_accum"]["losses"]):
        assert b == pytest.approx(a, rel=1e-4)


def test_fsdp_shards_model_state(runs):
    ddp = runs["ddp"]["state_bytes_per_rank"]
    fsdp = runs["fsdp"]["state_bytes_per_rank"]
    assert ddp["grads"] > 0 and ddp["optimizer"] > 0
    ratio = fsdp["total"] / ddp["total"]
    assert 0.45 < ratio < 0.6, ratio  # ~1/2 on 2 ranks (padding makes it slightly > 0.5)


def test_fsdp_meta_init_trains(runs):
    """Meta-device construction materializes only local shards and still trains."""
    losses = runs["fsdp_meta"]["losses"]
    assert all(l == l for l in losses)  # no NaN
    assert runs["fsdp_meta"]["state_bytes_per_rank"]["total"] == runs["fsdp"]["state_bytes_per_rank"]["total"]

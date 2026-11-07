"""Correctness tests. Run on CPU (gloo) in ~1 minute: pytest -q tests/

1. Analytical parameter count == real module parameter count (used for MFU).
2. DDP and FSDP on 2 and 4 ranks reproduce the single-device loss curve when the
   global batch is held fixed (same tokens, same math, just distributed).
3. FSDP per-rank state (params + grads + Adam) is ~1/N of DDP's.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dlt.model import CONFIGS, GPT  # noqa: E402


@pytest.mark.parametrize("name", ["tiny", "mini", "gpt-125m", "gpt-350m"])
def test_param_count_matches(name):
    cfg = CONFIGS[name]
    with torch.device("meta"):
        m = GPT(cfg)
    assert m.num_params() == cfg.num_params()


def _run(strategy, world, out, global_bs=8, steps=8):
    cmd = (["torchrun", "--standalone", f"--nproc_per_node={world}"] if strategy != "single"
           else [sys.executable])
    cmd += ["-m", "dlt.train", "--strategy", strategy, "--model", "tiny", "--data", "stdlib",
            "--steps", str(steps), "--warmup-steps", "1", "--micro-batch", str(global_bs // world),
            "--threads", "1", "--quiet", "--out", str(out)]
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="")  # exact fp32 check on CPU
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True)
    with open(out) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return _run("single", 1, tmp_path_factory.mktemp("b") / "single.json")


@pytest.mark.parametrize("strategy,world", [("ddp", 2), ("fsdp", 2), ("ddp", 4), ("fsdp", 4)])
def test_loss_matches_single(baseline, tmp_path, strategy, world):
    rec = _run(strategy, world, tmp_path / "r.json")
    for a, b in zip(baseline["losses"], rec["losses"]):
        assert abs(a - b) < 2e-4, (baseline["losses"], rec["losses"])


def test_fsdp_shards_state(tmp_path):
    ddp = _run("ddp", 4, tmp_path / "ddp.json", steps=3)
    fsdp = _run("fsdp", 4, tmp_path / "fsdp.json", steps=3)
    d = ddp["per_rank_memory"][0]["state_bytes"]
    f = max(r["state_bytes"] for r in fsdp["per_rank_memory"])
    # Sharding pads uneven dims, so allow a little slack over the ideal 1/4.
    assert f < d / 4 * 1.10, (d, f)

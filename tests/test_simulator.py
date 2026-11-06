"""Sanity checks on the simulator's mechanics (not on its accuracy)."""
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlt.model import get_config, num_params  # noqa: E402
from dlt.simulator import GPUS, LINKS, RunSpec, simulate, t_allreduce, t_gather  # noqa: E402

CFG = get_config("gpt-125m")


def test_model_state_is_16_bytes_per_param_sharded():
    P = num_params(CFG)
    for n in (1, 2, 4):
        assert simulate(RunSpec(CFG, "ddp", n, 8))["state_bytes"] == 16 * P
        assert simulate(RunSpec(CFG, "fsdp", n, 8))["state_bytes"] == pytest.approx(16 * P / n)


def test_ring_collective_costs():
    link = replace(LINKS["nvlink3"], alpha_us=0.0)
    b = 1e9
    assert t_allreduce(b, 4, link) == pytest.approx(2 * 3 / 4 * b / (link.busbw_gbs * 1e9))
    assert t_gather(b, 4, link) == pytest.approx(t_allreduce(b, 4, link) / 2)


def test_ddp_last_bucket_is_the_tied_embedding():
    out = simulate(RunSpec(CFG, "ddp", 4, 8))
    ar = [e for e in out["comm_events"] if e[2].startswith("all-reduce")]
    wte_mb = CFG.vocab_size * CFG.n_embd * 4 / 2**20
    assert float(ar[-1][2].split()[1][:-2]) >= wte_mb - 1  # last bucket holds the whole embedding
    embed_bwd_end = max(e for s, e, lab in out["compute_events"] if lab == "embed bwd")
    assert ar[-1][0] >= embed_bwd_end - 1e-12  # cannot start before backward finishes


def test_efficiency_improves_with_bandwidth():
    effs = []
    single = simulate(RunSpec(CFG, "single", 1, 8))["tokens_per_s"]
    for bw in (5, 20, 230):
        link = replace(LINKS["nvlink3"], busbw_gbs=bw)
        effs.append(simulate(RunSpec(CFG, "ddp", 4, 8, link=link))["tokens_per_s"] / (4 * single))
    assert effs == sorted(effs) and effs[-1] <= 1.0


def test_no_grad_sync_removes_ddp_comm():
    assert simulate(RunSpec(CFG, "ddp", 4, 8, grad_sync=False))["comm_s"] == 0.0


def test_fsdp_fits_models_ddp_cannot():
    big = get_config("gpt-13b")
    gpu = GPUS["a100-sxm4-80gb"]
    assert not simulate(RunSpec(big, "ddp", 4, 1, gpu=gpu))["fits"]
    assert simulate(RunSpec(big, "fsdp", 4, 1, gpu=gpu))["fits"]

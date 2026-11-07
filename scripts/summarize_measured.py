"""Turn Measured GPU runs into the headline numbers (all labeled Measured).

  python scripts/summarize_measured.py --dir results/gpu
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlt.model import CONFIGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/gpu")
    ap.add_argument("--peak-tflops", type=float, default=989.0, help="bf16 dense peak: H100 SXM 989, A100 312")
    a = ap.parse_args()
    load = lambda n: json.load(open(os.path.join(a.dir, n)))
    out = {"label": "Measured"}
    for m in ("gpt-125m", "gpt-350m"):
        try:
            single = load(f"{m}_single_w1.json")
        except FileNotFoundError:
            continue
        cfg = CONFIGS[m]
        flops_tok = 6 * cfg.num_params() + 12 * cfg.n_layer * cfg.seq_len * cfg.d_model
        out[f"mfu_single_{m}"] = single["tokens_per_s"] * flops_tok / (a.peak_tflops * 1e12)
        for s in ("ddp", "fsdp"):
            for w in (2, 4):
                r = load(f"{m}_{s}_w{w}.json")
                out[f"{m}_{s}_w{w}_scaling_efficiency"] = r["tokens_per_s"] / (w * single["tokens_per_s"])
                if r.get("exposed_comm_s") is not None:
                    out[f"{m}_{s}_w{w}_exposed_comm_ms"] = r["exposed_comm_s"] * 1e3
        d4, f4 = load(f"{m}_ddp_w4.json"), load(f"{m}_fsdp_w4.json")
        pk = lambda r: max(x["cuda_peak_allocated"] for x in r["per_rank_memory"])
        st = lambda r: max(x["state_bytes"] for x in r["per_rank_memory"])
        out[f"{m}_peak_mem_cut_fsdp4_vs_ddp4"] = 1 - pk(f4) / pk(d4)
        out[f"{m}_state_mem_cut_fsdp4_vs_ddp4"] = 1 - st(f4) / st(d4)
    for s in ("ddp", "fsdp"):
        ok = [json.load(open(f)) for f in glob.glob(os.path.join(a.dir, f"maxprobe_{s}_*.json"))]
        ok = [r for r in ok if r["status"] == "ok"]
        if ok:
            best = max(ok, key=lambda r: r["params"])
            out[f"max_model_{s}_4gpu"] = best["params"]
    if "max_model_ddp_4gpu" in out and "max_model_fsdp_4gpu" in out:
        out["max_model_ratio"] = out["max_model_fsdp_4gpu"] / out["max_model_ddp_4gpu"]
    json.dump(out, open(os.path.join(a.dir, "summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

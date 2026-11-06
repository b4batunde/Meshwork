"""Run the whole study in the simulator (dlt/simulator.py). Everything this script
writes is SIMULATED and labeled as such.

    python bench/simulate.py                 # full study -> results/simulated/, README blocks
    python bench/simulate.py --no-readme
    python bench/simulate.py --compare results/scaling.jsonl   # simulated vs measured, after GPU runs

Steps
  1. calibrate eta_gemm on ONE published single-GPU measurement
  2. validate against the other published measurements (reported, never tuned)
  3. simulate every run in the study (same records/tags as bench/run_scaling.sh)
  4. extra experiments: timelines, interconnect sweep, tied-embedding bottleneck,
     bf16 gradient communication, bucket size, platforms, sensitivity
  5. feed the records through bench/analyze.py --mode simulated
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from dlt.model import GPTConfig, MODEL_CONFIGS, get_config, num_params  # noqa: E402
from dlt.simulator import GPUS, LINKS, GiB, RunSpec, SimParams, simulate  # noqa: E402

OUT = ROOT / "results" / "simulated"
PRIMARY = dict(gpu="a100-sxm4-80gb", link="nvlink3", nop2p="pcie4-host")
STUDY = dict(models=["gpt-125m", "gpt-350m"], gpus=[1, 2, 4], mb=8, global_mb=32, seq=1024,
             sweep_batches=[1, 2, 4, 8, 16, 32])
COLORS = {"ddp": "#2a78d6", "fsdp": "#eb6834", "compute": "#86b6ef", "comm": "#184f95",
          "opt": "#a3a29b", "ideal": "#a3a29b"}

# Published measurements by others, used to calibrate (1) and validate (rest).
SRC_SCHWARTZ = "https://jjschwartz.github.io/blog/scaling-gpt2.html"
SRC_LLMC = "https://github.com/karpathy/llm.c/discussions/481"
ANCHORS = [
    dict(id="calib", role="calibration", what="GPT-2 124M, 1x H100, PyTorch", metric="MFU", value=0.36,
         spec=dict(model="gpt-125m", strategy="single", n=1, mb=64, gpu="h100-sxm-80gb", link="nvlink4"),
         src=SRC_SCHWARTZ),
    dict(id="v1", role="validation", what="GPT-2 124M, 8x H100 DDP, PyTorch", metric="MFU", value=0.33,
         spec=dict(model="gpt-125m", strategy="ddp", n=8, mb=64, gpu="h100-sxm-80gb", link="nvlink4"),
         src=SRC_SCHWARTZ),
    dict(id="v2", role="validation", what="GPT-2 124M, 1x to 8x H100 DDP scaling efficiency (33% / 36% MFU)",
         metric="efficiency", value=0.33 / 0.36,
         spec=dict(model="gpt-125m", strategy="ddp", n=8, mb=64, gpu="h100-sxm-80gb", link="nvlink4"),
         src=SRC_SCHWARTZ),
    dict(id="v3", role="validation", what="GPT-2 1.5B, 8x H100 DDP, PyTorch (micro-batch not reported; 8 x16 accum assumed)",
         metric="MFU", value=0.4194,
         spec=dict(model="gpt2-xl", strategy="ddp", n=8, mb=8, accum=16, gpu="h100-sxm-80gb", link="nvlink4"),
         src=SRC_SCHWARTZ),
    dict(id="v4", role="bound", what="GPT-2 124M, 1x A100 40GB PCIe, llm.c (hand-fused CUDA, fp16): simulated eager PyTorch should be lower",
         metric="MFU", value=0.49,
         spec=dict(model="gpt-125m", strategy="single", n=1, mb=64, gpu="a100-sxm4-40gb", link="nvlink3"),
         src=SRC_LLMC),
]

EXTRA_MODELS = {"gpt2-xl": GPTConfig(n_layer=48, n_head=25, n_embd=1600)}


def cfg_of(name, **kw):
    if name in EXTRA_MODELS:
        return replace(EXTRA_MODELS[name], **kw)
    return get_config(name, **kw)


def spec(model, strategy, n, mb, params, *, gpu=None, link=None, ckpt=False, **kw):
    return RunSpec(cfg_of(model, grad_ckpt=ckpt), strategy, n, mb, seq_len=STUDY["seq"], grad_ckpt=ckpt,
                   gpu=GPUS[gpu or PRIMARY["gpu"]], link=LINKS[link or PRIMARY["link"]], params=params, **kw)


def run_anchor(a, params):
    s = a["spec"]
    r = spec(s["model"], s["strategy"], s["n"], s["mb"], params, gpu=s["gpu"], link=s["link"],
             grad_accum=s.get("accum", 1))
    out = simulate(r)
    if a["metric"] == "MFU":
        return out["mfu"]
    single = simulate(replace(r, strategy="single", n=1))
    return out["tokens_per_s"] / (s["n"] * single["tokens_per_s"])


# ---------------------------------------------------------------------------
# 1-2. calibration and validation
# ---------------------------------------------------------------------------

def calibrate(base: SimParams) -> SimParams:
    """Scale eta_gemm (and eta_attn proportionally) so the calibration anchor matches."""
    a = ANCHORS[0]
    ratio = base.eta_attn / base.eta_gemm
    lo, hi = 0.2, 0.95
    for _ in range(40):
        mid = (lo + hi) / 2
        if run_anchor(a, replace(base, eta_gemm=mid, eta_attn=mid * ratio)) < a["value"]:
            lo = mid
        else:
            hi = mid
    return replace(base, eta_gemm=round(mid, 3), eta_attn=round(mid * ratio, 3))


def validation_rows(params):
    rows = []
    for a in ANCHORS:
        sim = run_anchor(a, params)
        rows.append(dict(id=a["id"], role=a["role"], what=a["what"], metric=a["metric"],
                         measured=a["value"], simulated=sim, err=sim - a["value"], src=a["src"]))
    return rows


# ---------------------------------------------------------------------------
# 3. the study, as train.py-schema records
# ---------------------------------------------------------------------------

def to_record(r: RunSpec, res: dict, tag: str, model: str) -> dict:
    ok = res["fits"]
    comp_busy = sum(e - s for s, e, lab in res["compute_events"] if lab not in ("wait", "interference"))
    rec = dict(source="simulated", tag=tag, model=model, n_params=num_params(r.cfg), strategy=r.strategy,
               world_size=r.n, micro_batch=r.micro_batch, grad_accum=r.grad_accum, seq_len=r.seq_len,
               global_batch_tokens=r.micro_batch * r.grad_accum * r.seq_len * r.n, dtype=r.gpu.half,
               grad_ckpt=r.grad_ckpt, no_grad_sync=not r.grad_sync, reshard_after_forward=r.reshard_after_forward,
               device=r.gpu.name, interconnect=r.link.name, status="ok" if ok else "oom")
    if ok:
        rec.update(step_time_median_s=res["step_s"], tokens_per_s=res["tokens_per_s"],
                   tokens_per_s_per_gpu=res["tokens_per_s"] / r.n, mfu=res["mfu"],
                   peak_mem_allocated_gb=res["peak_bytes"] / GiB,
                   state_bytes_per_rank=dict(total=res["state_bytes"]),
                   comm_profile=dict(comm_ms_per_step=res["comm_s"] * 1e3,
                                     comm_frac_of_kernel_time=res["comm_s"] / (res["comm_s"] + comp_busy)
                                     if res["comm_s"] else 0.0),
                   losses=[], final_loss=None)
    return rec


def study(params):
    rows = []
    add = lambda r, tag, m: rows.append(to_record(r, simulate(r), tag, m))
    for m in STUDY["models"]:
        add(spec(m, "single", 1, STUDY["mb"], params), "weak", m)
        add(spec(m, "single", 1, STUDY["global_mb"], params), "strong", m)
        for n in STUDY["gpus"]:
            for st in ("ddp", "fsdp"):
                add(spec(m, st, n, STUDY["mb"], params), "weak", m)
                add(spec(m, st, n, STUDY["global_mb"] // n, params), "strong", m)
                if n > 1:
                    add(spec(m, st, n, STUDY["mb"], params, grad_sync=False), "compute_only", m)
                    add(spec(m, st, n, STUDY["mb"], params), "profile", m)
                if n == max(STUDY["gpus"]):
                    add(spec(m, st, n, STUDY["mb"], params, link=PRIMARY["nop2p"]), "weak_nop2p", m)
    sweep = []
    for m in STUDY["models"]:
        for st in ("ddp", "fsdp"):
            for ck in (False, True):
                for mb in STUDY["sweep_batches"]:
                    r = spec(m, st, 4, mb, params, ckpt=ck)
                    sweep.append(to_record(r, simulate(r), "memsweep", m))
    return rows, sweep


def max_model(params, gpu, ckpt=False, n=4):
    ladder = [k for k in MODEL_CONFIGS if k != "tiny"]
    best = {}
    for st in ("ddp", "fsdp"):
        best[st] = None
        for m in ladder:
            if not simulate(spec(m, st, n, 1, params, gpu=gpu, ckpt=ckpt))["fits"]:
                break
            best[st] = m
    return {s: (m, num_params(get_config(m)) if m else 0) for s, m in best.items()}


# ---------------------------------------------------------------------------
# 4. extra experiments
# ---------------------------------------------------------------------------

def eff(r: RunSpec):
    out = simulate(r)
    single = simulate(replace(r, strategy="single", n=1, grad_accum=r.grad_accum))
    return out["tokens_per_s"] / (r.n * single["tokens_per_s"]), out


def experiments(params):
    E = {}
    m, n, mb = "gpt-125m", 4, STUDY["mb"]
    # Platforms
    plats = [("A100 80GB, NVLink", "a100-sxm4-80gb", "nvlink3"), ("A100 80GB, PCIe P2P", "a100-sxm4-80gb", "pcie4-p2p"),
             ("A100 80GB, P2P disabled", "a100-sxm4-80gb", "pcie4-host"), ("H100 80GB, NVLink", "h100-sxm-80gb", "nvlink4"),
             ("2x T4 (Kaggle-style), PCIe Gen3", "t4-16gb", "pcie3-host")]
    E["platforms"] = []
    for label, g, l in plats:
        nn = 2 if "T4" in label else 4
        for model in STUDY["models"]:
            row = dict(platform=label, model=model, n=nn)
            for st in ("ddp", "fsdp"):
                e, o = eff(spec(model, st, nn, mb if "T4" not in label else 4, params, gpu=g, link=l))
                row[st] = e
                row[st + "_exposed_ms"] = o["exposed_comm_s"] * 1e3
            E["platforms"].append(row)
    # Tied embedding: the last DDP bucket
    E["tied"] = []
    for model in STUDY["models"]:
        cfg = get_config(model)
        wte_mb = cfg.vocab_size * cfg.n_embd * 4 / 2**20
        for l in ("nvlink3", "pcie4-host"):
            r = spec(model, "ddp", n, mb, params, link=l)
            e, o = eff(r)
            ev = [x for x in o["comm_events"] if x[2].startswith("all-reduce")]
            last = ev[-1]
            e16, o16 = eff(replace(r, ddp_bf16_comm=True))
            E["tied"].append(dict(model=model, link=LINKS[l].name, wte_mb=wte_mb,
                                  wte_share=cfg.vocab_size * cfg.n_embd / num_params(cfg),
                                  last_bucket_ms=(last[1] - last[0]) * 1e3,
                                  exposed_ms=o["exposed_comm_s"] * 1e3,
                                  last_share_of_exposed=min(1.0, (last[1] - last[0]) * 1e3 / max(1e-9, o["exposed_comm_s"] * 1e3)),
                                  step_ms=o["step_s"] * 1e3, eff=e, eff_bf16=e16,
                                  exposed_bf16_ms=o16["exposed_comm_s"] * 1e3))
    # Bucket size
    E["buckets"] = []
    for l in ("nvlink3", "pcie4-host"):
        for bmb in (1, 5, 25, 100):
            p2 = replace(params, bucket_mb=bmb)
            e, o = eff(spec("gpt-350m", "ddp", n, mb, p2, link=l))
            E["buckets"].append(dict(link=LINKS[l].name, bucket_mb=bmb, eff=e, exposed_ms=o["exposed_comm_s"] * 1e3))
    # FSDP reshard_after_forward
    E["reshard"] = []
    for model in STUDY["models"]:
        for l in ("nvlink3", "pcie4-host"):
            row = dict(model=model, link=LINKS[l].name)
            for rs in (True, False):
                r = spec(model, "fsdp", n, mb, params, link=l, reshard_after_forward=rs)
                e, o = eff(r)
                peak = o["peak_bytes"] / GiB
                if not rs:  # keeping blocks gathered: every block's bf16 params stay resident
                    cfg = get_config(model)
                    peak += (cfg.n_layer - 2) * (12 * cfg.n_embd ** 2) * 2 / GiB
                row["reshard" if rs else "keep"] = dict(eff=e, peak=peak)
            E["reshard"].append(row)
    # Bandwidth sweep for the plot
    E["bw_sweep"] = []
    bws = [2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 230, 360]
    for model in STUDY["models"]:
        for st in ("ddp", "fsdp"):
            for bw in bws:
                l = replace(LINKS["nvlink3"], busbw_gbs=bw, alpha_us=15 if bw >= 100 else 25)
                r = RunSpec(get_config(model), st, n, mb, link=l, gpu=GPUS[PRIMARY["gpu"]], params=params)
                e, _ = eff(r)
                E["bw_sweep"].append(dict(model=model, strategy=st, bw=bw, eff=e))
    # Sensitivity of the headline numbers (gpt-350m, 4 GPUs, NVLink)
    base_d, _ = eff(spec("gpt-350m", "ddp", n, mb, params))
    base_f, _ = eff(spec("gpt-350m", "fsdp", n, mb, params))
    knobs = [("NCCL bus bandwidth", "link", (0.5, 1.5)), ("GEMM efficiency", "eta_gemm", (0.8, 1.2)),
             ("Overlap interference", "interference", (0.0, 2.0)), ("Straggler jitter", "straggler_cv", (0.0, 2.0)),
             ("Per-collective latency", "alpha", (0.5, 3.0))]
    E["sensitivity"] = []
    for label, k, (lo, hi) in knobs:
        vals = {}
        for tag, f in (("lo", lo), ("hi", hi)):
            p2, link = params, LINKS["nvlink3"]
            if k == "link":
                link = replace(link, busbw_gbs=link.busbw_gbs * f)
            elif k == "alpha":
                link = replace(link, alpha_us=link.alpha_us * f)
            elif k == "eta_gemm":
                p2 = replace(params, eta_gemm=params.eta_gemm * f, eta_attn=params.eta_attn * f)
            else:
                p2 = replace(params, **{k: getattr(params, k) * f})
            r = RunSpec(get_config("gpt-350m"), "ddp", n, mb, gpu=GPUS[PRIMARY["gpu"]], link=link, params=p2)
            vals[tag] = (eff(r)[0], eff(replace(r, strategy="fsdp"))[0])
        E["sensitivity"].append(dict(knob=label, range=f"x{lo} to x{hi}", ddp=(vals["lo"][0], vals["hi"][0]),
                                     fsdp=(vals["lo"][1], vals["hi"][1])))
    E["sens_base"] = (base_d, base_f)
    # Max model size on 40 GB and 80 GB, with and without checkpointing
    E["maxsize"] = []
    for g in ("a100-sxm4-40gb", "a100-sxm4-80gb"):
        for ck in (False, True):
            mm = max_model(params, g, ckpt=ck)
            E["maxsize"].append(dict(gpu=GPUS[g].name, ckpt=ck, ddp=mm["ddp"], fsdp=mm["fsdp"],
                                     ratio=mm["fsdp"][1] / mm["ddp"][1] if mm["ddp"][1] else None))
    # Timelines for the plot
    E["timelines"] = {}
    for st in ("ddp", "fsdp"):
        for l in ("nvlink3", "pcie4-host"):
            o = simulate(spec("gpt-125m", st, n, mb, params, link=l))
            E["timelines"][(st, l)] = o
    return E


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style(ax, title, xl, yl):
    ax.set_title(title + " (simulated)", loc="left", fontsize=11, color="#1a1a19", pad=8)
    ax.set_xlabel(xl, color="#52514c")
    ax.set_ylabel(yl, color="#52514c")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors="#52514c")
    ax.set_axisbelow(True)


def plot_timelines(E, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(4, 1, figsize=(12, 7.5), sharex=False)
    for ax, ((st, l), o) in zip(axes, E["timelines"].items()):
        for s0, e0, lab in o["compute_events"]:
            if lab == "wait":
                continue
            c = {"optimizer": COLORS["opt"], "interference": "#e6e5e0"}.get(lab, COLORS["compute"])
            ax.barh(1, (e0 - s0) * 1e3, left=s0 * 1e3, height=0.6, color=c, edgecolor="white", linewidth=0.5)
        for s0, e0, lab in o["comm_events"]:
            ax.barh(0, (e0 - s0) * 1e3, left=s0 * 1e3, height=0.6, color=COLORS["comm"], edgecolor="white", linewidth=0.5)
        ax.set_yticks([0, 1], ["NCCL", "Compute"])
        ax.set_xlim(0, o["step_s"] * 1e3 * 1.02)
        tag = "NVLink" if l == "nvlink3" else "P2P disabled"
        _style(ax, f"gpt-125m {st.upper()} x4, {tag}: {o['step_s']*1e3:.0f} ms/step, "
                   f"{o['exposed_comm_s']*1e3:.1f} ms exposed comm", "ms" if ax is axes[-1] else "", "")
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    fig.legend(handles=[Patch(color=COLORS["compute"], label="Forward / backward kernels"),
                        Patch(color=COLORS["opt"], label="Optimizer"),
                        Patch(color="#e6e5e0", label="Slowdown from overlapped NCCL"),
                        Patch(color=COLORS["comm"], label="NCCL collectives")],
               loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bw_sweep(E, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, model in zip(axes, STUDY["models"]):
        for st, ls in (("ddp", "-"), ("fsdp", "-")):
            pts = sorted((r["bw"], r["eff"] * 100) for r in E["bw_sweep"] if r["model"] == model and r["strategy"] == st)
            ax.plot(*zip(*pts), ls, marker="o", ms=5, lw=2, color=COLORS[st], label=st.upper(),
                    markeredgecolor="white", markeredgewidth=1)
        for name, bw in (("P2P off", 10), ("PCIe P2P", 20), ("NVLink3", 230), ("NVLink4", 360)):
            ax.axvline(bw, color="#c3c2b7", lw=1, ls=":")
            ax.text(bw, 3, name, rotation=90, fontsize=8, color="#52514c", ha="right", va="bottom")
        ax.axhline(100, color=COLORS["ideal"], lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(0, 108)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        _style(ax, f"{model}, 4 GPUs: efficiency vs bandwidth", "NCCL bus bandwidth (GB/s, log)",
               "Weak-scaling efficiency (%)" if ax is axes[0] else "")
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x):
    return f"{x:.0%}" if x is not None else "n/a"


def simulator_md(params, val, E):
    P = PRIMARY
    L = ["# Simulator: method, calibration, validation and experiments", "",
         "Every number on this page is **Simulated** by `dlt/simulator.py` unless it is in a column marked "
         "**Measured (published)**, which are other people's measurements used only to calibrate and check the simulator.", "",
         f"Primary platform: **4x {GPUS[P['gpu']].name}, {LINKS[P['link']].name}** (the ASU Sol job's target).", "",
         "## Method", "",
         "- **Compute:** every kernel of a training step (LayerNorm, 4 GEMMs, flash attention, GELU, residual adds, "
         "autocast casts, embedding, LM head, loss, grad clipping, AdamW) is costed with a roofline: "
         "`max(FLOPs / (peak x eta), bytes / (HBM x eta_mem), t_min)`.",
         "- **Communication:** NCCL ring collectives, `alpha + 2(N-1)/N x bytes / busbw` (all-reduce) and "
         "`alpha + (N-1)/N x bytes / busbw` (all-gather, reduce-scatter).",
         "- **Scheduling:** compute and NCCL run on separate streams. DDP all-reduces 25 MB fp32 buckets in reverse "
         "parameter order as gradients become ready; FSDP2 all-gathers each block in bf16 (prefetching one ahead), "
         "reduce-scatters fp32 gradients after each block's backward and shards the optimizer step. Overlapped "
         "communication slows compute (`interference`), and every sync waits for the slowest rank (`straggler_cv`).",
         "- **Memory:** model state is exact (16 B/param, divided by N for FSDP, and checked against Measured CPU runs); "
         "activations follow Korthikanti et al. 2022 (34·s·b·h bytes/layer, no s² term with flash attention); "
         "logits b·s·V·10 bytes; plus autocast weight copies or FSDP's unsharded buffers.", "",
         "## Parameters", "",
         "| Parameter | Value | Basis |", "|---|---|---|"]
    L += [f"| A100 peak / HBM | 312 TFLOPS / 1.56 to 2.04 TB/s | NVIDIA datasheet |",
          f"| H100 SXM peak / HBM | 989 TFLOPS / 3.35 TB/s | NVIDIA datasheet |",
          f"| NCCL bus bandwidth, NVLink 3 | {LINKS['nvlink3'].busbw_gbs} GB/s | ~85% of the ~270 GB/s P2P bandwidth measured on DGX A100 |",
          f"| NCCL bus bandwidth, NVLink 4 | {LINKS['nvlink4'].busbw_gbs} GB/s | ~370 GB/s/direction achieved (ASTRA-sim HGX-H100 validation) |",
          f"| NCCL bus bandwidth, P2P disabled | {LINKS['pcie4-host'].busbw_gbs} GB/s | P2P-disabled copies measured 15 to 17 GB/s; ring through host is slower (assumed) |",
          f"| Per-collective latency | {LINKS['nvlink3'].alpha_us} to {LINKS['pcie4-host'].alpha_us} us | assumed |",
          f"| GEMM efficiency `eta_gemm` | {params.eta_gemm} | **calibrated** (one published anchor, below) |",
          f"| Flash-attention efficiency | {params.eta_attn} | same scale factor as eta_gemm |",
          f"| HBM efficiency | {params.eta_mem} | assumed |",
          f"| Overlap interference | {params.interference} | assumed |",
          f"| Straggler jitter (CV) | {params.straggler_cv} | assumed |",
          f"| AdamW traffic | {params.adam_bytes:.0f} B/param | sum of the foreach kernels' reads and writes |",
          f"| Activations | {params.act_coeff:.0f}·s·b·h B/layer | Korthikanti et al. 2022 |", "",
          "## Calibration and validation", "",
          "One published measurement sets `eta_gemm`. The rest are compared, never tuned to.", "",
          "| Role | Setup | Metric | Measured (published) | Simulated | Error | Source |", "|---|---|---|---|---|---|---|"]
    for v in val:
        L.append(f"| {v['role']} | {v['what']} | {v['metric']} | {v['measured']:.1%} | {v['simulated']:.1%} | "
                 f"{v['err']*100:+.1f} pts | [link]({v['src']}) |")
    L += ["", "Read this table as the simulator's error bar. Where the published runs used torch.compile, fused "
          "kernels or different batch sizes than assumed, part of the error is configuration, not physics.", ""]

    L += ["## Platforms (simulated, micro-batch 8; T4 uses 4 and fp16)", "",
          "| Platform | Model | GPUs | DDP efficiency | FSDP efficiency | DDP exposed comm (ms) | FSDP exposed comm (ms) |",
          "|---|---|---|---|---|---|---|"]
    for r in E["platforms"]:
        L.append(f"| {r['platform']} | {r['model']} | {r['n']} | {pct(r['ddp'])} | {pct(r['fsdp'])} | "
                 f"{r['ddp_exposed_ms']:.1f} | {r['fsdp_exposed_ms']:.1f} |")
    L += ["", "![bandwidth sweep](bw_sweep.png)", "",
          "## Bottleneck: the tied embedding gradient arrives last", "",
          "`wte` is the first entry of `model.parameters()`, so DDP puts it in the **last** bucket, and because "
          "it is tied to the LM head its gradient is only complete after the embedding backward, the final kernel "
          "of the backward pass. Its all-reduce therefore can never overlap with compute.", "",
          "| Model | Interconnect | wte size (fp32) | Share of params | Last-bucket all-reduce | Total exposed comm | Step | Efficiency | Efficiency with bf16 grad comm |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in E["tied"]:
        L.append(f"| {r['model']} | {r['link']} | {r['wte_mb']:.0f} MB | {r['wte_share']:.0%} | {r['last_bucket_ms']:.1f} ms | "
                 f"{r['exposed_ms']:.1f} ms | {r['step_ms']:.0f} ms | {pct(r['eff'])} | {pct(r['eff_bf16'])} |")
    L += ["", "`--ddp-bf16-comm` (a real flag in train.py, using PyTorch's `bf16_compress_hook`) halves every "
          "all-reduce, including this one.", "",
          "## DDP bucket size (gpt-350m, 4 GPUs)", "",
          "| Interconnect | bucket_cap_mb | Efficiency | Exposed comm (ms) |", "|---|---|---|---|"]
    for r in E["buckets"]:
        L.append(f"| {r['link']} | {r['bucket_mb']} | {pct(r['eff'])} | {r['exposed_ms']:.1f} |")
    L += ["", "## FSDP: reshard after forward vs keep gathered (4 GPUs)", "",
          "| Model | Interconnect | Efficiency (reshard) | Peak GiB (reshard) | Efficiency (keep) | Peak GiB (keep) |",
          "|---|---|---|---|---|---|"]
    for r in E["reshard"]:
        L.append(f"| {r['model']} | {r['link']} | {pct(r['reshard']['eff'])} | {r['reshard']['peak']:.1f} | "
                 f"{pct(r['keep']['eff'])} | {r['keep']['peak']:.1f} |")
    L += ["", "## Largest trainable model (4 GPUs, micro-batch 1)", "",
          "| GPU | Act. ckpt | DDP | FSDP | Ratio |", "|---|---|---|---|---|"]
    for r in E["maxsize"]:
        d, f = r["ddp"], r["fsdp"]
        L.append(f"| {r['gpu']} | {'yes' if r['ckpt'] else 'no'} | {d[0]} ({d[1]/1e9:.2f}B) | {f[0]} ({f[1]/1e9:.2f}B) | "
                 f"{r['ratio']:.1f}x |" if r["ratio"] else f"| {r['gpu']} | {r['ckpt']} | none | {f[0]} | n/a |")
    L += ["", "The ladder stops at 20B; an FSDP entry at the top of the ladder is a lower bound.", "",
          "## Sensitivity (gpt-350m, 4x A100, NVLink)", "",
          f"Baseline simulated efficiency: DDP {pct(E['sens_base'][0])}, FSDP {pct(E['sens_base'][1])}.", "",
          "| Assumption varied | Range | DDP efficiency | FSDP efficiency |", "|---|---|---|---|"]
    for r in E["sensitivity"]:
        L.append(f"| {r['knob']} | {r['range']} | {pct(r['ddp'][0])} to {pct(r['ddp'][1])} | "
                 f"{pct(r['fsdp'][0])} to {pct(r['fsdp'][1])} |")
    L += ["", "## Step timelines", "", "![timelines](timeline.png)", "",
          "## Limitations", "",
          "- No kernel fusion or torch.compile modeled; eager PyTorch only.",
          "- FSDP2's all-gather and reduce-scatter streams are modeled as one FIFO channel (pessimistic for FSDP).",
          "- Bandwidth does not degrade for mid-size messages beyond the fixed latency term.",
          "- CPU dispatch is modeled as a throughput limit, not as a timeline; Python-side stalls are not.",
          "- Memory ignores allocator fragmentation beyond a fixed 1.5 GiB headroom.",
          "- Every one of these is testable: run `bench/run_scaling.sh` on GPUs and `python bench/simulate.py --compare`."]
    return "\n".join(L) + "\n"


def readme_block(params, val, E):
    v = {x["id"]: x for x in val}
    tied = {(r["model"], r["link"]): r for r in E["tied"]}
    t125n = tied[("gpt-125m", LINKS["nvlink3"].name)]
    t125p = tied[("gpt-125m", LINKS["pcie4-host"].name)]
    mm = {(r["gpu"], r["ckpt"]): r for r in E["maxsize"]}
    m80 = mm[(GPUS["a100-sxm4-80gb"].name, False)]
    L = [f"Simulator: [`dlt/simulator.py`](dlt/simulator.py), full method, calibration and experiments in "
         f"[`results/simulated/SIMULATOR.md`](results/simulated/SIMULATOR.md).", "",
         "**Calibration and validation** (published measurements by others vs this simulator):", "",
         "| Setup | Measured (published) | Simulated | Error |", "|---|---|---|---|"]
    for x in val:
        L.append(f"| {x['what']} ({x['metric']}, {x['role']}) | {x['measured']:.1%} | {x['simulated']:.1%} | {x['err']*100:+.1f} pts |")
    L += ["", "**Simulated: the tied-embedding bottleneck.** For gpt-125m, the embedding matrix is "
          f"{t125n['wte_share']:.0%} of all parameters and its gradient is the last one ready, so its "
          f"{t125n['wte_mb']:.0f} MB all-reduce is always exposed: {t125n['last_bucket_ms']:.1f} ms on NVLink, "
          f"but {t125p['last_bucket_ms']:.0f} ms with P2P disabled ({t125p['last_share_of_exposed']:.0%} of all exposed "
          f"communication). bf16 gradient communication lifts simulated efficiency there from {pct(t125p['eff'])} to "
          f"{pct(t125p['eff_bf16'])}.", "",
          f"**Simulated: largest trainable model on 4x {m80['gpu']}:** DDP {m80['ddp'][0]}, FSDP {m80['fsdp'][0]} "
          f"({m80['ratio']:.1f}x).", "",
          "![timelines](results/simulated/timeline.png)", "![bandwidth sweep](results/simulated/bw_sweep.png)", ""]
    return "\n".join(L)


def write_block(readme: Path, key: str, body: str):
    text = readme.read_text()
    start, end = f"<!-- {key}:START -->", f"<!-- {key}:END -->"
    if start not in text:
        print(f"warning: README has no {start}")
        return
    pre, rest = text.split(start, 1)
    readme.write_text(pre + start + "\n" + body.rstrip() + "\n" + end + rest.split(end, 1)[1])


def compare(measured_path, params, link_key):
    """Simulate each measured run's exact configuration and print simulated vs measured."""
    meas = [json.loads(l) for l in Path(measured_path).read_text().splitlines() if l.strip()]
    gpu_for = lambda dev: next((k for k, g in GPUS.items() if g.name.split("-")[0].split()[-1] in dev),
                               PRIMARY["gpu"])
    print(f"Simulated vs Measured (interconnect assumed: {LINKS[link_key].name})\n")
    print("| Model | Strategy | GPUs | mb | Tag | Measured tok/s | Simulated tok/s | Error | Measured GiB | Simulated GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in meas:
        if m.get("status") != "ok" or m.get("tag") in ("profile", "memsweep", "maxsize"):
            continue
        r = RunSpec(cfg_of(m["model"], grad_ckpt=m.get("grad_ckpt", False)), m["strategy"], m["world_size"],
                    m["micro_batch"], seq_len=m["seq_len"], grad_ckpt=m.get("grad_ckpt", False),
                    grad_sync=not m.get("no_grad_sync", False), grad_accum=m.get("grad_accum", 1),
                    gpu=GPUS[gpu_for(m.get("device", ""))], link=LINKS[link_key], params=params)
        sim = simulate(r)
        mg = m.get("peak_mem_allocated_gb")
        print(f"| {m['model']} | {m['strategy']} | {m['world_size']} | {m['micro_batch']} | {m['tag']} | "
              f"{m['tokens_per_s']:,.0f} | {sim['tokens_per_s']:,.0f} | {sim['tokens_per_s']/m['tokens_per_s']-1:+.0%} | "
              f"{mg if mg is None else round(mg, 1)} | {sim['peak_bytes']/GiB:.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-readme", action="store_true")
    ap.add_argument("--compare", default=None, help="measured runs jsonl to compare against")
    ap.add_argument("--link", default=PRIMARY["link"], choices=list(LINKS),
                    help="interconnect of the measured machine (for --compare)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    params = calibrate(SimParams())
    if args.compare:
        compare(args.compare, params, args.link)
        return
    val = validation_rows(params)
    rows, sweep = study(params)
    (OUT / "scaling.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (OUT / "memsweep.jsonl").write_text("\n".join(json.dumps(r) for r in sweep) + "\n")
    mm = max_model(params, PRIMARY["gpu"])
    summ = dict(nproc=4, grad_ckpt=False, largest=mm, source="simulated", gpu=GPUS[PRIMARY["gpu"]].name)
    (OUT / "maxsize_summary.jsonl").write_text(json.dumps(summ) + "\n")
    (OUT / "calibration.json").write_text(json.dumps(dict(params=params.__dict__, validation=val), indent=2))

    E = experiments(params)
    plot_timelines(E, OUT / "timeline.png")
    plot_bw_sweep(E, OUT / "bw_sweep.png")
    (OUT / "SIMULATOR.md").write_text(simulator_md(params, val, E))

    import analyze
    analyze.MODE = "simulated"
    a = analyze.analyze(analyze.load([OUT / "scaling.jsonl", OUT / "memsweep.jsonl"]),
                        analyze.load([OUT / "maxsize_summary.jsonl"]))
    rep = OUT / "report"
    rep.mkdir(exist_ok=True)
    files = analyze.plot(a, rep)
    analyze.report_md(a, files, rep)
    if not args.no_readme:
        analyze.update_readme(a, files, rep, ROOT / "README.md")
        write_block(ROOT / "README.md", "SIMULATOR", readme_block(params, val, E))
    print((OUT / "SIMULATOR.md").read_text())


if __name__ == "__main__":
    main()

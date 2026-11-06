"""Turn raw run records into the report: tables, plots and the headline.

    python bench/analyze.py                         # reads results/scaling.jsonl (+ maxsize)
    python bench/analyze.py --runs results/a.jsonl results/b.jsonl --outdir results/report

Metrics
-------
weak-scaling efficiency  = tok/s on N GPUs / (N * tok/s of single-GPU baseline), same per-GPU batch
strong-scaling speedup   = step time of single-GPU at global batch G / step time on N GPUs at G
exposed comm overhead    = 1 - (compute-only step time / full step time)
memory reduction (FSDP)  = 1 - peak per-GPU memory FSDP / peak per-GPU memory DDP, same N
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlt.model import get_config, num_params  # noqa: E402

# Validated categorical slots (blue, orange) + neutral for the baseline/ideal.
COLORS = {"ddp": "#2a78d6", "fsdp": "#eb6834", "single": "#6b6a64", "ideal": "#a3a29b"}
LABELS = {"single": "Single GPU", "ddp": "DDP", "fsdp": "FSDP"}

# Every result is labeled either "measured" (real runs of train.py) or "simulated"
# (records from bench/simulate.py). The label appears in every title, table and headline.
MODE = "measured"
_VERBS = dict(reached="reached", was="was", hid="hid", dropped="dropped", exceed="exceed",
              are="are", achieved="Achieved", cut="cut", delivered="delivered")
TAG = {"measured": "Measured", "simulated": "Simulated"}


def V(key):
    return _VERBS[key]


def load(paths):
    rows = []
    for p in paths:
        p = Path(p)
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def mem_gb(r):
    """Measured peak GPU memory if we have it, else model-state bytes (CPU runs)."""
    if r.get("peak_mem_allocated_gb") is not None:
        return r["peak_mem_allocated_gb"], "peak allocated"
    return r["state_bytes_per_rank"]["total"] / 2**30, "model state (params+grads+Adam)"


def latest(rows, **kw):
    hit = [r for r in rows if all(r.get(k) == v for k, v in kw.items()) and r.get("status") == "ok"]
    return hit[-1] if hit else None


def analyze(rows, maxsize_rows, headline_model=None):
    models = list(dict.fromkeys(r["model"] for r in rows if r.get("tag") == "weak"))
    gpus = sorted({r["world_size"] for r in rows if r.get("tag") == "weak"})
    out = {"models": models, "gpus": gpus, "weak": [], "strong": [], "comm": [], "mem": [],
           "interconnect": [], "memsweep": []}

    for m in models:
        base = latest(rows, model=m, strategy="single", tag="weak")
        sbase = latest(rows, model=m, strategy="single", tag="strong")
        if base:
            g, kind = mem_gb(base)
            out["mem"].append(dict(model=m, strategy="single", n=1, gb=g, kind=kind,
                                   state_gb=base["state_bytes_per_rank"]["total"] / 2**30))
        for strat in ("ddp", "fsdp"):
            for n in gpus:
                r = latest(rows, model=m, strategy=strat, world_size=n, tag="weak")
                if r and base:
                    eff = r["tokens_per_s"] / (n * base["tokens_per_s"])
                    out["weak"].append(dict(model=m, strategy=strat, n=n, tok_s=r["tokens_per_s"],
                                            base_tok_s=base["tokens_per_s"], eff=eff, mfu=r.get("mfu"),
                                            step_ms=r["step_time_median_s"] * 1e3))
                if r:
                    g, kind = mem_gb(r)
                    out["mem"].append(dict(model=m, strategy=strat, n=n, gb=g, kind=kind,
                                           state_gb=r["state_bytes_per_rank"]["total"] / 2**30))
                s = latest(rows, model=m, strategy=strat, world_size=n, tag="strong")
                if s and sbase and sbase["global_batch_tokens"] == s["global_batch_tokens"]:
                    sp = sbase["step_time_median_s"] / s["step_time_median_s"]
                    out["strong"].append(dict(model=m, strategy=strat, n=n, speedup=sp, eff=sp / n))
                c = latest(rows, model=m, strategy=strat, world_size=n, tag="compute_only")
                if r and c:
                    full, comp = r["step_time_median_s"] * 1e3, c["step_time_median_s"] * 1e3
                    prof = latest(rows, model=m, strategy=strat, world_size=n, tag="profile")
                    cp = (prof or {}).get("comm_profile") or {}
                    exposed_ms = max(0.0, full - comp)
                    comm_ms = cp.get("comm_ms_per_step")
                    # Comm kernels that ran while compute kept going are "hidden".
                    hidden = (min(1.0, max(0.0, 1 - exposed_ms / comm_ms)) if comm_ms else None)
                    out["comm"].append(dict(model=m, strategy=strat, n=n, full_ms=full, compute_ms=comp,
                                            exposed=exposed_ms / full, exposed_ms=exposed_ms,
                                            comm_kernel_ms=comm_ms, hidden_frac=hidden,
                                            kernel_comm_frac=cp.get("comm_frac_of_kernel_time")))
                # Interconnect: same run with NVLink/P2P disabled (NCCL_P2P_DISABLE=1).
                np2p = latest(rows, model=m, strategy=strat, world_size=n, tag="weak_nop2p")
                if np2p and r and base:
                    out["interconnect"].append(dict(
                        model=m, strategy=strat, n=n,
                        eff_p2p=r["tokens_per_s"] / (n * base["tokens_per_s"]),
                        eff_nop2p=np2p["tokens_per_s"] / (n * base["tokens_per_s"]),
                        slowdown=r["tokens_per_s"] / np2p["tokens_per_s"]))

    # Memory sweep: peak memory vs micro-batch, with and without activation checkpointing.
    for r in rows:
        if r.get("tag") == "memsweep":
            st = r.get("state_bytes_per_rank", {}).get("total", 0) / 2**30
            pk = r.get("peak_mem_allocated_gb")
            out["memsweep"].append(dict(model=r["model"], strategy=r["strategy"], n=r["world_size"],
                                        mb=r["micro_batch"], ckpt=r["grad_ckpt"], status=r["status"],
                                        peak_gb=pk, state_gb=st,
                                        act_gb=(pk - st) if pk is not None else None))
    out["crossover"] = activation_crossover(out["memsweep"])

    # Max model size
    z = None
    summ = [r for r in maxsize_rows if "largest" in r]
    if summ:
        s = summ[-1]
        (dm, dp), (fm, fp) = s["largest"].get("ddp", (None, 0)), s["largest"].get("fsdp", (None, 0))
        if dp:
            z = dict(ddp_model=dm, ddp_params=dp, fsdp_model=fm, fsdp_params=fp, ratio=fp / dp,
                     nproc=s["nproc"], grad_ckpt=s["grad_ckpt"])
    out["maxsize"] = z
    out["headline"] = headline(out, headline_model)
    out["findings"] = findings(out)
    return out


def headline(a, model=None):
    if not a["models"] or not a["gpus"]:
        return None
    m = model or max(a["models"], key=lambda x: num_params(get_config(x)))
    n = max(a["gpus"])
    w = {r["strategy"]: r for r in a["weak"] if r["model"] == m and r["n"] == n}
    mem = {r["strategy"]: r for r in a["mem"] if r["model"] == m and r["n"] == n}
    if not w:
        return None
    best = w.get("ddp") or max(w.values(), key=lambda r: r["eff"])
    lead = f"{V('achieved')} " if MODE == "measured" else ""
    parts = [f"{lead}{best['eff']:.0%} weak-scaling efficiency on {n} GPUs "
             f"({LABELS[best['strategy']]}, {m})"]
    if "ddp" in w and "fsdp" in w:
        parts[-1] += f", with FSDP at {w['fsdp']['tok_s'] / w['ddp']['tok_s']:.2f}x DDP's throughput"
    y = None
    if "ddp" in mem and "fsdp" in mem:
        y = 1 - mem["fsdp"]["gb"] / mem["ddp"]["gb"]
        parts.append(f"FSDP {V('cut')} per-GPU memory {y:.0%}")
    if a["maxsize"]:
        z = a["maxsize"]
        parts[-1] += f", enabling {z['ratio']:.1f}x larger models ({z['ddp_model']} -> {z['fsdp_model']})"
    return {"text": "; ".join(parts) + ".", "X": best["eff"], "Y": y,
            "Z": a["maxsize"]["ratio"] if a["maxsize"] else None}


def activation_crossover(sweep):
    """For each (model, strategy, n, ckpt): the smallest micro-batch at which
    activations + temporaries use more memory than the (sharded) model state.
    Past this point FSDP's sharding stops being the dominant memory lever."""
    groups = defaultdict(list)
    for r in sweep:
        if r["status"] == "ok" and r["act_gb"] is not None:
            groups[(r["model"], r["strategy"], r["n"], r["ckpt"])].append(r)
    res = []
    for (m, s, n, ck), rs in groups.items():
        rs.sort(key=lambda r: r["mb"])
        hit = next((r for r in rs if r["act_gb"] > r["state_gb"]), None)
        res.append(dict(model=m, strategy=s, n=n, ckpt=ck, crossover_mb=hit["mb"] if hit else None,
                        max_mb_tested=rs[-1]["mb"],
                        act_share_at_max=rs[-1]["act_gb"] / rs[-1]["peak_gb"] if rs[-1]["peak_gb"] else None))
    return res


def findings(a):
    """Data-driven first draft of the 'why' section. Every sentence quotes a
    measured number; edit the wording, keep the numbers."""
    F = []
    by = lambda rows, **kw: [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    n = max(a["gpus"]) if a["gpus"] else None
    for m in a["models"]:
        d = by(a["weak"], model=m, strategy="ddp", n=n)
        f = by(a["weak"], model=m, strategy="fsdp", n=n)
        if d and f:
            d, f = d[0], f[0]
            gap = 1 - f["tok_s"] / d["tok_s"]
            cd = by(a["comm"], model=m, strategy="ddp", n=n)
            cf = by(a["comm"], model=m, strategy="fsdp", n=n)
            rel = (f"{abs(gap):.0%} {'less' if gap > 0 else 'more'} throughput than DDP"
                   if abs(gap) >= 0.005 else "about the same throughput as DDP")
            s = (f"**{m}, {n} GPUs:** DDP {V('reached')} {d['eff']:.0%} weak-scaling efficiency vs "
                 f"{f['eff']:.0%} for FSDP (FSDP {V('delivered')} {rel}).")
            if cd and cf:
                s += (f" Exposed communication {V('was')} {cd[0]['exposed_ms']:.1f} ms/step for DDP vs "
                      f"{cf[0]['exposed_ms']:.1f} ms/step for FSDP")
                if cd[0]["hidden_frac"] is not None and cf[0]["hidden_frac"] is not None:
                    s += (f"; DDP {V('hid')} {cd[0]['hidden_frac']:.0%} of its NCCL time behind compute, "
                          f"FSDP {cf[0]['hidden_frac']:.0%}")
                s += "."
            F.append(s)
    for r in a["interconnect"]:
        F.append(f"**Interconnect ({r['model']}, {LABELS[r['strategy']]} x{r['n']}):** disabling NVLink/P2P "
                 f"{V('dropped')} efficiency from {r['eff_p2p']:.0%} to {r['eff_nop2p']:.0%} "
                 f"({r['slowdown']:.2f}x slower).")
    for c in a["crossover"]:
        if c["strategy"] == "fsdp":
            tail = (f"activations {V('exceed')} sharded model state from micro-batch {c['crossover_mb']}"
                    if c["crossover_mb"] else f"model state still dominates up to micro-batch {c['max_mb_tested']}")
            share = f"{c['act_share_at_max']:.0%}" if c["act_share_at_max"] is not None else "n/a"
            F.append(f"**Activation memory ({c['model']}, FSDP x{c['n']}{', ckpt' if c['ckpt'] else ''}):** "
                     f"{tail}; at micro-batch {c['max_mb_tested']} activations {V('are')} {share} of peak.")
    return F


def theoretical_state_gb(model, n, strategy):
    """Mixed-precision Adam: fp32 params (4) + fp32 grads (4) + Adam m,v (8) = 16 B/param.
    FSDP shards all of it across n ranks."""
    p = num_params(get_config(model))
    return 16 * p / (n if strategy == "fsdp" else 1) / 2**30


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def style(ax, title, xlabel, ylabel):
    title += f" ({TAG[MODE].lower()})"
    ax.set_title(title, loc="left", fontsize=12, color="#1a1a19", pad=10)
    ax.set_xlabel(xlabel, color="#52514c")
    ax.set_ylabel(ylabel, color="#52514c")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514c")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot(a, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    for m in a["models"]:
        rows = [r for r in a["weak"] if r["model"] == m]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        # Throughput vs GPUs with ideal-linear reference
        ax = axes[0]
        base = rows[0]["base_tok_s"]
        ns = sorted({r["n"] for r in rows} | {1})
        ax.plot(ns, [base * n / 1e3 for n in ns], "--", color=COLORS["ideal"], lw=1.5, label="Ideal (linear)")
        for strat in ("ddp", "fsdp"):
            pts = sorted((r["n"], r["tok_s"] / 1e3) for r in rows if r["strategy"] == strat)
            if pts:
                ax.plot(*zip(*pts), "-o", color=COLORS[strat], lw=2, ms=8, label=LABELS[strat],
                        markeredgecolor="white", markeredgewidth=2)
                ax.annotate(f"{pts[-1][1]:,.0f}k", pts[-1], textcoords="offset points", xytext=(8, 4 if strat == "fsdp" else -12),
                            color="#1a1a19", fontsize=9)
        ax.set_xticks(ns)
        style(ax, f"{m}: throughput", "GPUs", "Tokens/s (thousands)")
        ax.legend(frameon=False)
        # Efficiency bars
        ax = axes[1]
        width = 0.38
        xs = range(len(ns))
        for i, strat in enumerate(("ddp", "fsdp")):
            effs = [next((r["eff"] for r in rows if r["strategy"] == strat and r["n"] == n), 0) for n in ns]
            pos = [x + (i - 0.5) * width for x in xs]
            bars = ax.bar(pos, [e * 100 for e in effs], width - 0.03, color=COLORS[strat], label=LABELS[strat])
            for b, e in zip(bars, effs):
                if e:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{e:.0%}",
                            ha="center", fontsize=9, color="#1a1a19")
        ax.axhline(100, color=COLORS["ideal"], ls="--", lw=1.2)
        ax.set_xticks(list(xs), [str(n) for n in ns])
        ax.set_ylim(0, 115)
        style(ax, f"{m}: weak-scaling efficiency", "GPUs", "% of linear")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        fig.tight_layout()
        f = outdir / f"scaling_{m}.png"
        fig.savefig(f, dpi=160)
        plt.close(fig)
        files.append(f)

    # Memory per GPU
    if a["mem"]:
        models = a["models"]
        fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.2), squeeze=False)
        for ax, m in zip(axes[0], models):
            rows = [r for r in a["mem"] if r["model"] == m]
            ns = sorted({r["n"] for r in rows if r["strategy"] != "single"})
            width = 0.38
            for i, strat in enumerate(("ddp", "fsdp")):
                vals = [next((r["gb"] for r in rows if r["strategy"] == strat and r["n"] == n), 0) for n in ns]
                pos = [x + (i - 0.5) * width for x in range(len(ns))]
                bars = ax.bar(pos, vals, width - 0.03, color=COLORS[strat], label=LABELS[strat])
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}", ha="center",
                            va="bottom", fontsize=9, color="#1a1a19")
            ax.set_xticks(range(len(ns)), [str(n) for n in ns])
            kind = rows[0]["kind"]
            style(ax, f"{m}: per-GPU memory", "GPUs", f"GiB ({kind})")
            ax.legend(frameon=False)
        fig.tight_layout()
        f = outdir / "memory_per_gpu.png"
        fig.savefig(f, dpi=160)
        plt.close(fig)
        files.append(f)

    # Step-time breakdown: compute vs exposed communication
    if a["comm"]:
        rows = a["comm"]
        labels = [f"{r['model']}\n{LABELS[r['strategy']]} x{r['n']}" for r in rows]
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(rows)), 4.2))
        comp = [r["compute_ms"] for r in rows]
        comm = [r["full_ms"] - r["compute_ms"] if r["full_ms"] > r["compute_ms"] else 0 for r in rows]
        ax.bar(range(len(rows)), comp, 0.6, color="#86b6ef", label="Compute")
        ax.bar(range(len(rows)), comm, 0.6, bottom=comp, color="#184f95", label="Exposed communication",
               edgecolor="white", linewidth=2)
        for i, r in enumerate(rows):
            ax.text(i, r["full_ms"], f"{r['exposed']:.0%}", ha="center", va="bottom", fontsize=9, color="#1a1a19")
        ax.set_xticks(range(len(rows)), labels, fontsize=8)
        style(ax, "Where step time goes (label = exposed comm share)", "", "ms per step")
        ax.legend(frameon=False)
        fig.tight_layout()
        f = outdir / "step_breakdown.png"
        fig.savefig(f, dpi=160)
        plt.close(fig)
        files.append(f)

    # Peak memory vs micro-batch: where activations overtake sharded model state
    sweep = [r for r in a["memsweep"] if r["status"] == "ok" and r["peak_gb"] is not None]
    if sweep:
        models = sorted({r["model"] for r in sweep}, key=lambda m: num_params(get_config(m)))
        fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.2), squeeze=False)
        for ax, m in zip(axes[0], models):
            for strat in ("ddp", "fsdp"):
                for ck, ls in ((False, "-"), (True, ":")):
                    pts = sorted((r["mb"], r["peak_gb"]) for r in sweep
                                 if r["model"] == m and r["strategy"] == strat and r["ckpt"] == ck)
                    if pts:
                        ax.plot(*zip(*pts), ls, marker="o", color=COLORS[strat], lw=2, ms=7,
                                markeredgecolor="white", markeredgewidth=1.5,
                                label=f"{LABELS[strat]}{' + act. ckpt' if ck else ''}")
                st = [r["state_gb"] for r in sweep if r["model"] == m and r["strategy"] == strat]
                if st:
                    ax.axhline(st[0], color=COLORS[strat], lw=1, ls="--", alpha=0.6,
                               label=f"{LABELS[strat]} model state (params+grads+Adam)")
            mbs = sorted({r["mb"] for r in sweep if r["model"] == m})
            ax.set_xscale("log", base=2)
            ax.set_xticks(mbs, [str(b) for b in mbs])
            ax.minorticks_off()
            style(ax, f"{m}: peak memory vs micro-batch", "Micro-batch per GPU", "Peak GiB per GPU")
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        f = outdir / "memory_vs_batch.png"
        fig.savefig(f, dpi=160)
        plt.close(fig)
        files.append(f)
    return files


def results_md(a, files, img_prefix="", h="##"):
    """Results body shared by REPORT.md and the README section."""
    L = []
    if a["headline"]:
        L += [f"**{TAG[MODE]}:** {a['headline']['text']}", ""]
    if a["findings"]:
        L += [f"{h} Key findings ({TAG[MODE].lower()})", ""]
        L += [f"- {f}" for f in a["findings"]] + [""]
    L += [f"{h} Weak scaling (fixed per-GPU batch)", "",
          "| Model | Strategy | GPUs | Tokens/s | Step (ms) | Efficiency | MFU |", "|---|---|---|---|---|---|---|"]
    for r in a["weak"]:
        mfu = f"{r['mfu']:.1%}" if r["mfu"] else "n/a"
        L.append(f"| {r['model']} | {LABELS[r['strategy']]} | {r['n']} | {r['tok_s']:,.0f} | "
                 f"{r['step_ms']:.1f} | {r['eff']:.1%} | {mfu} |")
    if a["strong"]:
        L += ["", f"{h} Strong scaling (fixed global batch)", "",
              "| Model | Strategy | GPUs | Speedup | Efficiency |", "|---|---|---|---|---|"]
        for r in a["strong"]:
            L.append(f"| {r['model']} | {LABELS[r['strategy']]} | {r['n']} | {r['speedup']:.2f}x | {r['eff']:.1%} |")
    L += ["", f"{h} Per-GPU memory", "",
          f"| Model | Strategy | GPUs | {TAG[MODE]} peak (GiB) | Model state (GiB) | Theory, 16 B/param (GiB) |",
          "|---|---|---|---|---|---|"]
    for r in a["mem"]:
        th = theoretical_state_gb(r["model"], r["n"], r["strategy"])
        L.append(f"| {r['model']} | {LABELS[r['strategy']]} | {r['n']} | {r['gb']:.2f} | "
                 f"{r['state_gb']:.2f} | {th:.2f} |")
    if a["comm"]:
        L += ["", f"{h} Communication: how much is hidden behind backward?", "",
              "| Model | Strategy | GPUs | Full step (ms) | Compute-only (ms) | Exposed comm | NCCL kernel time (ms) | Hidden |",
              "|---|---|---|---|---|---|---|---|"]
        fmt = lambda v, f: format(v, f) if v is not None else "n/a"
        for r in a["comm"]:
            L.append(f"| {r['model']} | {LABELS[r['strategy']]} | {r['n']} | {r['full_ms']:.1f} | "
                     f"{r['compute_ms']:.1f} | {r['exposed']:.1%} | {fmt(r['comm_kernel_ms'], '.1f')} | "
                     f"{fmt(r['hidden_frac'], '.0%')} |")
    if a["interconnect"]:
        L += ["", f"{h} NVLink vs no-P2P", "",
              "| Model | Strategy | GPUs | Efficiency (P2P on) | Efficiency (P2P off) | Slowdown |", "|---|---|---|---|---|---|"]
        for r in a["interconnect"]:
            L.append(f"| {r['model']} | {LABELS[r['strategy']]} | {r['n']} | {r['eff_p2p']:.0%} | "
                     f"{r['eff_nop2p']:.0%} | {r['slowdown']:.2f}x |")
    if a["crossover"]:
        L += ["", f"{h} Where activation memory takes over", "",
              "| Model | Strategy | GPUs | Act. ckpt | Activations > model state from micro-batch | Activation share at largest batch |",
              "|---|---|---|---|---|---|"]
        for c in a["crossover"]:
            sh = f"{c['act_share_at_max']:.0%}" if c["act_share_at_max"] is not None else "n/a"
            L.append(f"| {c['model']} | {LABELS[c['strategy']]} | {c['n']} | {'yes' if c['ckpt'] else 'no'} | "
                     f"{c['crossover_mb'] or 'not reached (max ' + str(c['max_mb_tested']) + ')'} | {sh} |")
    if a["maxsize"]:
        z = a["maxsize"]
        L += ["", f"{h} Largest trainable model", "",
              f"On {z['nproc']} GPUs (micro-batch 1{', activation checkpointing' if z['grad_ckpt'] else ''}): "
              f"DDP fits **{z['ddp_model']}** ({z['ddp_params']/1e6:,.0f}M), FSDP fits **{z['fsdp_model']}** "
              f"({z['fsdp_params']/1e6:,.0f}M), **{z['ratio']:.1f}x** larger."]
    L += [""] + [f"![{f.stem}]({img_prefix}{f.name})" for f in files]
    return "\n".join(L) + "\n"


def report_md(a, files, outdir: Path):
    body = results_md(a, files)
    (outdir / "REPORT.md").write_text("# Distributed LLM Training: Results\n\n" + body)
    (outdir / "summary.json").write_text(json.dumps(a, indent=2))
    return body


def update_readme(a, files, outdir: Path, readme: Path):
    """Replace this mode's results and headline blocks in the README."""
    text = readme.read_text()
    prefix = os.path.relpath(outdir, readme.parent) + "/"
    sfx = "_SIM" if MODE == "simulated" else ""
    blocks = {f"RESULTS{sfx}": results_md(a, files, img_prefix=prefix, h="###")}
    if a["headline"]:
        blocks[f"HEADLINE{sfx}"] = f"> **{TAG[MODE]}:** {a['headline']['text']}"
    for key, body in blocks.items():
        start, end = f"<!-- {key}:START -->", f"<!-- {key}:END -->"
        if start in text and end in text:
            pre, rest = text.split(start, 1)
            text = pre + start + "\n" + body.rstrip() + "\n" + end + rest.split(end, 1)[1]
        else:
            print(f"warning: README has no {start} block")
    readme.write_text(text)
    print(f"updated {readme}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["results/scaling.jsonl", "results/memsweep.jsonl"])
    p.add_argument("--maxsize", default="results/maxsize_summary.jsonl")
    p.add_argument("--outdir", default="results/report")
    p.add_argument("--headline-model", default=None, help="model for the headline (default: largest studied)")
    p.add_argument("--update-readme", action="store_true", help="write results into README.md")
    p.add_argument("--mode", choices=["measured", "simulated"], default="measured")
    args = p.parse_args()
    global MODE
    MODE = args.mode
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    a = analyze(load(args.runs), load([args.maxsize]), args.headline_model)
    files = plot(a, outdir)
    print(report_md(a, files, outdir))
    if args.update_readme:
        update_readme(a, files, outdir, ROOT / "README.md")


if __name__ == "__main__":
    main()

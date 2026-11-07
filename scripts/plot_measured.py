"""Figures from the measured GPU runs.

  python scripts/plot_measured.py --dir results/gpu
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class P:  # shared chart style
    BLUE, ORANGE, YELLOW = "#2a78d6", "#eb6834", "#eda100"
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    STRAT = {"ddp": BLUE, "fsdp": ORANGE}

    @staticmethod
    def _save(fig, d, name):
        os.makedirs(d, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(d, name), dpi=160)
        plt.close(fig)


plt.rcParams.update({
    "figure.facecolor": P.SURF, "axes.facecolor": P.SURF, "savefig.facecolor": P.SURF,
    "axes.edgecolor": P.GRID, "axes.labelcolor": P.INK2, "xtick.color": P.INK2, "ytick.color": P.INK2,
    "axes.grid": True, "grid.color": P.GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": P.INK,
    "legend.frameon": False, "lines.linewidth": 2,
})

GiB = 2 ** 30
MODELS = ("gpt-125m", "gpt-350m")


def load(d, n):
    with open(os.path.join(d, n)) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/gpu")
    ap.add_argument("--fig-dir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    gpu = open(os.path.join(a.dir, "gpus.csv")).read().splitlines()[1].split(",")[0]

    # ---- 1. scaling efficiency
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, m in zip(axes, MODELS):
        single = load(a.dir, f"{m}_single_w1.json")["tokens_per_s"]
        for st in ("ddp", "fsdp"):
            xs, ys = [1], [100.0]
            for w in (2, 4):
                r = load(a.dir, f"{m}_{st}_w{w}.json")
                xs.append(w); ys.append(100 * r["tokens_per_s"] / (w * single))
            ax.plot(xs, ys, color=P.STRAT[st], marker="o", ms=6, label=st.upper())
            dy = 6 if st == "ddp" else -6
            ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                        va="center", color=P.INK2, fontsize=9)
        ax.set_title(m); ax.set_xticks([1, 2, 4]); ax.set_xlabel("GPUs"); ax.set_xlim(0.8, 4.6)
    axes[0].set_ylabel("Scaling efficiency vs 1 GPU (%)"); axes[0].set_ylim(50, 102)
    axes[-1].legend(loc="lower left", fontsize=8.5)
    fig.suptitle(f"Weak scaling on {gpu}, NVLink, 8 seq/GPU",
                 color=P.INK, fontweight="bold", fontsize=11)
    P._save(fig, a.fig_dir, "scaling_efficiency.png")

    # ---- 2. step-time breakdown at 4 GPUs (Measured): baseline, wrapper overhead, exposed comm
    fig, ax = plt.subplots(figsize=(9, 3.4))
    labels, base, over, comm = [], [], [], []
    for m in MODELS:
        s = load(a.dir, f"{m}_single_w1.json")["step_time_s"] * 1e3
        for st in ("ddp", "fsdp"):
            r = load(a.dir, f"{m}_{st}_w4.json")
            t, ns = r["step_time_s"] * 1e3, r["nosync_step_time_s"] * 1e3
            labels.append(f"{m} {st.upper()} ×4"); base.append(s)
            over.append(max(0, ns - s)); comm.append(max(0, t - ns))
    y = range(len(labels))
    ax.barh(y, base, color=P.BLUE, label="single-GPU step", edgecolor=P.SURF, linewidth=1.5)
    ax.barh(y, over, left=base, color=P.YELLOW, label="wrapper overhead (DDP/FSDP, no sync)",
            edgecolor=P.SURF, linewidth=1.5)
    ax.barh(y, comm, left=[b + o for b, o in zip(base, over)], color=P.ORANGE,
            label="exposed gradient sync", edgecolor=P.SURF, linewidth=1.5)
    for i in y:
        tot = base[i] + over[i] + comm[i]
        ax.text(tot + 1, i, f"{tot:.1f} ms", va="center", color=P.INK2, fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("ms per step"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    ax.set_title(f"Where the 4-GPU step time goes on {gpu}", pad=26)
    P._save(fig, a.fig_dir, "step_breakdown.png")

    # ---- 3. per-GPU memory at 4 GPUs (Measured CUDA peak vs states)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    labels, states, rest = [], [], []
    for m in MODELS:
        for st in ("ddp", "fsdp"):
            r = load(a.dir, f"{m}_{st}_w4.json")
            mem = max(r["per_rank_memory"], key=lambda x: x["cuda_peak_allocated"])
            labels.append(f"{m}\n{st.upper()} ×4")
            states.append(mem["state_bytes"] / GiB)
            rest.append((mem["cuda_peak_allocated"] - mem["state_bytes"]) / GiB)
    ax.bar(labels, states, color=P.BLUE, width=0.6, label="params + grads + Adam", edgecolor=P.SURF, linewidth=1.5)
    ax.bar(labels, rest, bottom=states, color=P.ORANGE, width=0.6, label="activations, logits, buffers",
           edgecolor=P.SURF, linewidth=1.5)
    for i, (s_, r_) in enumerate(zip(states, rest)):
        ax.text(i, s_ + r_ + 0.3, f"{s_ + r_:.1f}", ha="center", color=P.INK2, fontsize=9)
    ax.set_ylabel("Peak GiB per GPU"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(s + r for s, r in zip(states, rest)) * 1.15)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(f"Per-GPU memory, 4 GPUs, 8 seq/GPU")
    P._save(fig, a.fig_dir, "memory.png")

    # ---- 4. max model ladder (Measured)
    ladder = ["gpt-760m", "gpt-1.3b", "gpt-2.7b", "gpt-6.7b", "gpt-13b"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.38
    for k, st in enumerate(("ddp", "fsdp")):
        for i, m in enumerate(ladder):
            p = os.path.join(a.dir, f"maxprobe_{st}_{m}.json")
            x = i + (k - 0.5) * w
            if os.path.exists(p) and load(a.dir, os.path.basename(p))["status"] == "ok":
                r = load(a.dir, os.path.basename(p))
                pk = max(x_["cuda_peak_allocated"] for x_ in r["per_rank_memory"]) / GiB
                ax.bar(x, pk, w - 0.04, color=P.STRAT[st], label=st.upper() if i == 0 else None)
                ax.text(x, pk + 1, f"{pk:.0f}", ha="center", color=P.INK2, fontsize=8.5)
            else:
                ax.text(x, 2, "OOM", ha="center", color=P.STRAT[st], fontsize=9, fontweight="bold",
                        rotation=90, va="bottom")
    ax.axhline(79.2, color=P.INK2, lw=1, ls="--")
    ax.text(len(ladder) - 0.5, 80.5, "80 GB HBM", ha="right", color=P.INK2, fontsize=8.5)
    ax.set_xticks(range(len(ladder))); ax.set_xticklabels([m.replace("gpt-", "") for m in ladder])
    ax.set_xlabel("Model size (params)"); ax.set_ylabel("Peak GiB per GPU")
    ax.set_ylim(0, 90); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    ax.set_title(f"Largest trainable model on 4× {gpu.replace('NVIDIA ', '')},, activation checkpointing")
    P._save(fig, a.fig_dir, "max_model.png")
    print("wrote", a.fig_dir)


if __name__ == "__main__":
    main()

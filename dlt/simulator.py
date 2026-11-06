"""Event-driven performance simulator for single-GPU, DDP and FSDP2 training of
the GPT model in dlt/model.py. Everything it produces is SIMULATED, not measured.

How it works
------------
1. Compute: every kernel of one training step (per transformer block: LayerNorms,
   four GEMMs, flash attention, GELU, residual adds, autocast casts; plus the
   embedding, LM head, loss, grad clipping and AdamW) is costed with a roofline:

       t_kernel = max(FLOPs / (peak * eta_gemm|eta_attn),  bytes / (HBM_BW * eta_mem),  t_min)

   so small models become memory/launch bound and big ones compute bound on their own.

2. Communication: NCCL ring collectives with an alpha-beta model on "bus bandwidth":
       all-reduce            t = alpha + 2(N-1)/N * bytes / busbw
       all-gather / RS       t = alpha +  (N-1)/N * bytes / busbw

3. Timeline: compute and communication run on separate streams. The simulator
   replays the real scheduling rules of each strategy:
     DDP   gradients land in 25 MB fp32 buckets in reverse parameter order; a bucket's
           all-reduce starts when its last gradient is ready and the comm stream is free.
           (The first bucket is 1 MB, as in PyTorch.) The tied embedding is the FIRST
           parameter, so it sits in the LAST bucket and is ready only after the
           embedding backward, the very end of the backward pass.
     FSDP2 per-block bf16 all-gather (prefetched one block ahead in forward and in
           backward), fp32 reduce-scatter after each block's backward, root unit
           (embeddings + head) gathered once and not resharded, sharded optimizer.
   Communication that overlaps compute slows compute by `interference` x overlap
   (NCCL kernels occupy SMs and HBM bandwidth).

4. Memory: exact model-state bytes (the same quantities train.py measures), plus
   activations (Korthikanti et al. 2022), logits, autocast weight copies and
   FSDP's unsharded buffers.

The free parameters live in SimParams. `eta_gemm` is calibrated on ONE published
single-GPU number (see bench/simulate.py); everything else is a datasheet value
or a documented assumption, and the other published numbers are used only to
validate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from dlt.model import GPTConfig, flops_per_token, num_params

GiB = 2**30


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPU:
    name: str
    peak_tflops: float      # dense bf16/fp16 tensor-core peak (datasheet)
    hbm_gbs: float          # HBM bandwidth, GB/s (datasheet)
    mem_gib: float          # usable device memory
    half: str = "bf16"      # T4 has no bf16


@dataclass(frozen=True)
class Link:
    name: str
    busbw_gbs: float        # achievable NCCL bus bandwidth, large messages
    alpha_us: float         # fixed cost per collective (launch + sync + ring latency)


GPUS = {
    "a100-sxm4-40gb": GPU("A100-SXM4-40GB", 312, 1555, 40),
    "a100-sxm4-80gb": GPU("A100-SXM4-80GB", 312, 2039, 80),
    "h100-sxm-80gb":  GPU("H100-SXM5-80GB", 989, 3350, 80),
    "t4-16gb":        GPU("Tesla T4 16GB", 65, 320, 15, half="fp16"),
}

LINKS = {
    # 8x A100 NVSwitch P2P measured at ~270 GB/s unidirectional; NCCL ring reaches ~85% of it.
    "nvlink3":    Link("NVLink 3 (A100, NVSwitch)", 230, 15),
    # H100: 450 GB/s/dir spec, ~370 GB/s achieved per direction (ASTRA-sim HGX-H100 validation).
    "nvlink4":    Link("NVLink 4 (H100, NVSwitch)", 360, 12),
    # PCIe Gen4 x16 with peer-to-peer through a PCIe switch.
    "pcie4-p2p":  Link("PCIe Gen4, P2P", 20, 20),
    # NCCL_P2P_DISABLE=1: staged through host memory (P2P-disabled copies measured ~15-17 GB/s).
    "pcie4-host": Link("PCIe Gen4, P2P disabled (via host)", 10, 30),
    # Kaggle-style 2x T4 on PCIe Gen3 without P2P.
    "pcie3-host": Link("PCIe Gen3, via host", 5, 35),
}


@dataclass(frozen=True)
class SimParams:
    eta_gemm: float = 0.55        # achieved/peak for large bf16 GEMMs in eager PyTorch (CALIBRATED)
    eta_attn: float = 0.40        # flash-attention kernels (SDPA) relative to peak
    eta_mem: float = 0.75         # achieved/peak HBM bandwidth for elementwise kernels
    t_min_us: float = 4.0         # minimum duration of any kernel
    cpu_op_us: float = 9.0        # eager-mode CPU dispatch cost per kernel launch
    fsdp_unit_cpu_us: float = 120.0  # FSDP2 Python hook cost per unit per pass
    interference: float = 0.15    # compute slowdown per ms of overlapped NCCL time
    adam_bytes: float = 64.0      # HBM bytes per (local) param, foreach AdamW (lerp, mul+addcmul, sqrt/div/add, addcdiv, decay)
    bucket_mb: float = 25.0       # DDP bucket_cap_mb
    first_bucket_mb: float = 1.0  # DDP's small first bucket
    act_coeff: float = 34.0       # activation bytes per layer = act_coeff * s * b * h (Korthikanti et al.)
    logits_bytes: float = 10.0    # bytes per logit at peak: bf16 logits + fp32 copy + fp32 grad
    straggler_cv: float = 0.015   # per-rank step-time jitter (coefficient of variation); a sync waits for the slowest rank
    mem_overhead_gib: float = 1.5 # CUDA context + allocator fragmentation headroom for "fits?"


# ---------------------------------------------------------------------------
# Kernel cost model
# ---------------------------------------------------------------------------

class Cost:
    """Accumulates GPU time and kernel count for a sequence of kernels."""

    def __init__(self, gpu: GPU, p: SimParams):
        self.gpu, self.p = gpu, p
        self.t = 0.0
        self.kernels = 0

    def gemm(self, flops, bytes_):
        self._add(flops, bytes_, self.p.eta_gemm)

    def attn(self, flops, bytes_):
        self._add(flops, bytes_, self.p.eta_attn)

    def mem(self, bytes_):
        self._add(0.0, bytes_, 1.0)

    def _add(self, flops, bytes_, eta):
        tc = flops / (self.gpu.peak_tflops * 1e12 * eta) if flops else 0.0
        tm = bytes_ / (self.gpu.hbm_gbs * 1e9 * self.p.eta_mem)
        self.t += max(tc, tm, self.p.t_min_us * 1e-6)
        self.kernels += 1


def block_costs(cfg: GPTConfig, b: int, s: int, gpu: GPU, p: SimParams, *, autocast_casts: bool,
                ckpt: bool):
    """(forward_s, backward_s, kernels_fwd, kernels_bwd) for ONE transformer block."""
    h, n = cfg.n_embd, b * s
    W = 12 * h * h  # block weight elements

    def fwd(c: Cost):
        if autocast_casts:                      # fp32 -> bf16 weight copies (qkv, proj, fc, fc2)
            for k in (3, 1, 4, 4):
                c.mem(k * h * h * 6)
        c.mem(n * h * 6)                        # ln1 (fp32 residual in, bf16 out)
        c.gemm(2 * n * h * 3 * h, 2 * (n * h + 3 * h * h + 3 * n * h))      # qkv
        c.attn(2 * b * s * s * h, 2 * 4 * n * h)                            # causal flash attn
        c.gemm(2 * n * h * h, 2 * (2 * n * h + h * h))                      # attn proj
        c.mem(n * h * 10)                       # residual add (fp32 + bf16 -> fp32)
        c.mem(n * h * 6)                        # ln2
        c.gemm(2 * n * h * 4 * h, 2 * (n * h + 4 * h * h + 4 * n * h))     # fc
        c.mem(n * 4 * h * 4)                    # gelu
        c.gemm(2 * n * 4 * h * h, 2 * (4 * n * h + 4 * h * h + n * h))     # fc2
        c.mem(n * h * 10)                       # residual add

    f = Cost(gpu, p)
    fwd(f)
    bw = Cost(gpu, p)
    if ckpt:
        fwd(bw)                                 # recompute
    for k in (3, 1, 4, 4):                      # dgrad + wgrad per GEMM
        bw.gemm(2 * n * h * k * h, 2 * (n * h + k * h * h + k * n * h))
        bw.gemm(2 * n * h * k * h, 2 * (n * h + k * h * h + k * n * h))
    bw.attn(2.5 * 2 * b * s * s * h, 2 * 8 * n * h)                      # flash attn backward
    bw.mem(n * 4 * h * 6)                       # gelu backward
    for _ in range(2):
        bw.mem(n * h * 14)                      # layernorm backward
        bw.mem(n * h * 8)                       # residual grad accumulate
    if autocast_casts:
        bw.mem(W * 6)                           # bf16 wgrad -> fp32 .grad
    return f.t, bw.t, f.kernels, bw.kernels


def head_costs(cfg: GPTConfig, b: int, s: int, gpu: GPU, p: SimParams, *, autocast_casts: bool):
    """Embedding + final LayerNorm + tied LM head + cross-entropy."""
    h, V, n = cfg.n_embd, cfg.vocab_size, b * s
    e = Cost(gpu, p)
    e.mem(n * h * 12)                           # token + position embedding, add
    f = Cost(gpu, p)
    f.mem(n * h * 6)                            # ln_f
    if autocast_casts:
        f.mem(V * h * 6)                        # cast tied weight to bf16
    f.gemm(2 * n * h * V, 2 * (n * h + V * h + n * V))   # lm_head
    f.mem(n * V * 6)                            # logits.float()
    f.mem(n * V * 8)                            # log_softmax + nll
    bw = Cost(gpu, p)
    bw.mem(n * V * 8)                           # softmax grad
    bw.mem(n * V * 6)                           # fp32 -> bf16 grad
    bw.gemm(2 * n * h * V, 2 * (n * V + V * h + n * h))  # dgrad
    bw.gemm(2 * n * h * V, 2 * (n * V + n * h + V * h))  # wgrad
    if autocast_casts:
        bw.mem(V * h * 6)
    bw.mem(n * h * 14)                          # ln_f backward
    emb = Cost(gpu, p)                          # embedding backward (end of backward pass)
    emb.mem(V * h * 4 + n * h * 16)             # zero + scatter-add into wte grad, wpe grad
    return e, f, bw, emb


def optimizer_cost(local_params: float, gpu: GPU, p: SimParams) -> Cost:
    c = Cost(gpu, p)
    c.mem(local_params * 12)                    # clip_grad_norm_: norm + scale
    c.mem(local_params * p.adam_bytes)          # foreach AdamW
    c.kernels += 30                             # foreach launches (approx.)
    return c


# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------

def t_allreduce(nbytes, n, link: Link):
    if n == 1:
        return 10e-6
    return link.alpha_us * 1e-6 + 2 * (n - 1) / n * nbytes / (link.busbw_gbs * 1e9)


def t_gather(nbytes, n, link: Link):
    """all-gather or reduce-scatter of a tensor whose FULL size is nbytes."""
    if n == 1:
        return 10e-6
    return link.alpha_us * 1e-6 + (n - 1) / n * nbytes / (link.busbw_gbs * 1e9)


EXPECTED_MAX_NORMAL = {2: 0.564, 3: 0.846, 4: 1.029, 8: 1.424, 16: 1.766}


class Stream:
    """A FIFO CUDA stream: work starts when issued AND the stream is free."""

    def __init__(self, name):
        self.name, self.free, self.events = name, 0.0, []

    def run(self, issue, dur, label):
        start = max(issue, self.free)
        self.free = start + dur
        self.events.append((start, self.free, label))
        return self.free


# ---------------------------------------------------------------------------
# Simulation of one optimizer step
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    cfg: GPTConfig
    strategy: str            # single | ddp | fsdp
    n: int
    micro_batch: int
    seq_len: int = 1024
    grad_ckpt: bool = False
    grad_sync: bool = True
    reshard_after_forward: bool = True
    grad_accum: int = 1
    ddp_bf16_comm: bool = False  # DDP bf16_compress_hook: all-reduce gradients in bf16
    gpu: GPU = GPUS["a100-sxm4-40gb"]
    link: Link = LINKS["nvlink3"]
    params: SimParams = field(default_factory=SimParams)


def param_layout(cfg: GPTConfig):
    """Parameter element counts in model.parameters() order (tied lm_head appears once)."""
    h, V, T = cfg.n_embd, cfg.vocab_size, cfg.block_size
    block = 12 * h * h + 2 * h
    return dict(wte=V * h, wpe=T * h, block=block, ln_f=h, n_layer=cfg.n_layer)


def simulate_step(r: RunSpec) -> dict:
    cfg, p, gpu, link, n = r.cfg, r.params, r.gpu, r.link, r.n
    b, s = r.micro_batch, r.seq_len
    lay = param_layout(cfg)
    L = cfg.n_layer
    autocast = r.strategy != "fsdp"     # FSDP casts via MixedPrecisionPolicy at all-gather time
    bf, bb, kf, kb = block_costs(cfg, b, s, gpu, p, autocast_casts=autocast, ckpt=r.grad_ckpt)
    ef, hf, hb, he = head_costs(cfg, b, s, gpu, p, autocast_casts=autocast)
    P = num_params(cfg)
    local_P = P / n if r.strategy == "fsdp" else P
    opt = optimizer_cost(local_P, gpu, p)

    comp, comm = Stream("compute"), Stream("comm")
    kernels = L * (kf + kb) + ef.kernels + hf.kernels + hb.kernels + he.kernels + opt.kernels
    cpu_extra = 0.0

    if r.strategy in ("single", "ddp"):
        t = comp.run(0, ef.t, "embed")              # embedding part of the head cost
        for i in range(L):
            t = comp.run(t, bf, f"fwd b{i}")
        t = comp.run(t, hf.t, "head+loss")
        t = comp.run(t, hb.t, "head bwd")
        ready = {"ln_f": t}
        for i in reversed(range(L)):
            t = comp.run(t, bb, f"bwd b{i}")
            ready[f"b{i}"] = t
        t = comp.run(t, he.t, "embed bwd")
        ready["wpe"] = ready["wte"] = t
        bwd_end = t

        if r.strategy == "ddp" and r.grad_sync:
            # Buckets in reverse model.parameters() order: ln_f, b{L-1}..b0, wpe, wte.
            order = [("ln_f", lay["ln_f"])] + [(f"b{i}", lay["block"]) for i in reversed(range(L))] \
                    + [("wpe", lay["wpe"]), ("wte", lay["wte"])]
            buckets, cur, cur_bytes, cap = [], [], 0.0, p.first_bucket_mb * 2**20
            for name, numel in order:
                cur.append(name)
                cur_bytes += numel * 4
                if cur_bytes >= cap:
                    buckets.append((cur, cur_bytes))
                    cur, cur_bytes, cap = [], 0.0, p.bucket_mb * 2**20
            if cur:
                buckets.append((cur, cur_bytes))
            for names, nbytes in buckets:
                if r.ddp_bf16_comm:
                    nbytes /= 2
                comm.run(max(ready[x] for x in names), t_allreduce(nbytes, n, link),
                         f"all-reduce {nbytes/2**20:.0f}MB")
            cpu_extra += 20e-6 * len(buckets)
    elif r.strategy == "fsdp":
        h = cfg.n_embd
        root_bf16 = (lay["wte"] + lay["wpe"] + lay["ln_f"]) * 2
        blk_bf16 = lay["block"] * 2
        blk_copy = Cost(gpu, p)                # copy-in (fp32 shard -> bf16) + copy-out (split)
        blk_copy.mem(lay["block"] / n * 6 + blk_bf16 * 2)
        rs_copy = Cost(gpu, p)                 # bf16 grads -> fp32 reduce-scatter input
        rs_copy.mem(lay["block"] * 6)
        kernels += L * 6 + 8
        cpu_extra += (L + 1) * 3 * p.fsdp_unit_cpu_us * 1e-6

        # ---- forward
        ag_root = comm.run(0, t_gather(root_bf16, n, link), "AG root")
        ag = {0: comm.run(0, t_gather(blk_bf16, n, link), "AG b0")}
        t = comp.run(ag_root, ef.t, "embed")
        for i in range(L):
            start = max(t, ag[i])
            if i + 1 < L:                       # prefetch next block while this one computes
                ag[i + 1] = comm.run(start, t_gather(blk_bf16, n, link), f"AG b{i+1}")
            t = comp.run(start, blk_copy.t + bf, f"fwd b{i}")
        t = comp.run(t, hf.t, "head+loss")
        t = comp.run(t, hb.t, "head bwd")
        # ---- backward
        if r.reshard_after_forward:
            agb = {L - 1: comm.run(t, t_gather(blk_bf16, n, link), f"AG(bwd) b{L-1}")}
        else:
            agb = {i: 0.0 for i in range(L)}
        for i in reversed(range(L)):
            start = max(t, agb[i])
            if r.reshard_after_forward and i - 1 >= 0:
                agb[i - 1] = comm.run(start, t_gather(blk_bf16, n, link), f"AG(bwd) b{i-1}")
            t = comp.run(start, (blk_copy.t if r.reshard_after_forward else 0) + bb + rs_copy.t, f"bwd b{i}")
            if r.grad_sync:
                comm.run(t, t_gather(lay["block"] * 4, n, link), f"RS b{i}")
        t = comp.run(t, he.t, "embed bwd")
        if r.grad_sync:
            comm.run(t, t_gather((lay["wte"] + lay["wpe"] + lay["ln_f"]) * 4, n, link), "RS root")
        bwd_end = t
    else:
        raise ValueError(r.strategy)

    # Overlapped communication steals SMs/HBM from compute.
    comm_busy = sum(e - st for st, e, _ in comm.events)
    overlapped = sum(max(0.0, min(e, bwd_end) - st) for st, e, _ in comm.events if st < bwd_end)
    slowdown = p.interference * overlapped
    bwd_end += slowdown
    comm_end = comm.free if comm.events else 0.0   # NCCL itself is not slowed; compute is
    sync_end = max(bwd_end, comm_end)
    step_gpu = sync_end + opt.t
    busy = sum(e - st for st, e, _ in comp.events)
    # Time the compute stream sat idle waiting for communication (all-gathers mid-pass,
    # or the gradient sync at the end): the "exposed" communication.
    exposed = max(0.0, sync_end - busy - slowdown)
    if slowdown > 0:
        comp.events.append((bwd_end - slowdown, bwd_end, "interference"))
    if sync_end > bwd_end:
        comp.events.append((bwd_end, sync_end, "wait"))
    comp.events.append((sync_end, step_gpu, "optimizer"))
    cpu = kernels * p.cpu_op_us * 1e-6 + cpu_extra
    step = max(step_gpu, cpu)

    # Gradient accumulation: the first (accum-1) micro-steps skip gradient sync and the optimizer.
    prefix = 0.0
    if r.grad_accum > 1:
        sub = simulate_step(replace(r, grad_accum=1, grad_sync=False, params=replace(p, straggler_cv=0.0)))
        prefix = (r.grad_accum - 1) * (sub["step_s"] - sub["optimizer_s"])
    step += prefix
    # Every sync waits for the slowest of n ranks: E[max of n] ~= mean + cv * E[max of n N(0,1)].
    if n > 1:
        step *= 1 + p.straggler_cv * EXPECTED_MAX_NORMAL.get(n, 1.54)

    tokens = b * s * n * r.grad_accum
    tok_s = tokens / step
    mfu = flops_per_token(cfg, s) * tok_s / n / (gpu.peak_tflops * 1e12)
    return dict(step_s=step, tokens_per_s=tok_s, mfu=mfu, comm_s=comm_busy, exposed_comm_s=exposed,
                interference_s=slowdown, cpu_bound=cpu > step_gpu, compute_events=comp.events, comm_events=comm.events,
                optimizer_s=opt.t)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def simulate_memory(r: RunSpec) -> dict:
    """Peak allocated bytes per GPU (what torch.cuda.max_memory_allocated reports)."""
    cfg, p, n = r.cfg, r.params, r.n
    b, s, h, V, L = r.micro_batch, r.seq_len, cfg.n_embd, cfg.vocab_size, cfg.n_layer
    P = num_params(cfg)
    lay = param_layout(cfg)
    if r.grad_ckpt:
        acts = L * 4 * s * b * h + p.act_coeff * s * b * h      # saved fp32 block inputs + 1 live layer
    else:
        acts = L * p.act_coeff * s * b * h
    acts += 4 * s * b * h                                          # embedding output
    logits = p.logits_bytes * s * b * V

    if r.strategy in ("single", "ddp"):
        params, grads, adam = 4 * P, 4 * P, 8 * P
        bf16_copies = 2 * P                                        # autocast weights saved for backward
        if r.strategy == "ddp":   # gradient buckets exist for the whole run
            peak = params + grads + adam + bf16_copies + acts + logits
        else:                     # grads appear during backward while activations are freed
            peak = max(params + adam + bf16_copies + acts + logits + 4 * lay["wte"],
                       params + grads + adam + 4 * P * 0.25)       # optimizer step + foreach temps
        state = params + grads + adam
    else:
        shard = P / n
        state = 16 * shard
        root_bf16 = (lay["wte"] + lay["wpe"] + lay["ln_f"]) * 2
        gathered = root_bf16 + 2 * lay["block"] * 2                # root + current + prefetched block
        bwd_bufs = lay["block"] * (2 + 4) + lay["wte"] * 6         # block grads bf16+fp32, wte grad
        peak = 12 * shard + gathered + acts + logits + max(bwd_bufs, 4 * shard)
        peak = max(peak, state + gathered + 4 * shard * 0.25)
    return dict(peak_bytes=peak, state_bytes=state, act_bytes=acts, logits_bytes=logits,
                fits=peak / GiB <= r.gpu.mem_gib - p.mem_overhead_gib)


def simulate(r: RunSpec) -> dict:
    out = simulate_step(r)
    out.update(simulate_memory(r))
    return out


def with_params(r: RunSpec, **kw) -> RunSpec:
    return replace(r, params=replace(r.params, **kw))

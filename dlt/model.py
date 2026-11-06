"""GPT-style decoder-only transformer (GPT-2 architecture, pre-LayerNorm).

Kept deliberately close to GPT-2 so that parameter counts and FLOP estimates
are directly comparable with published numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50304  # GPT-2 vocab (50257) padded to a multiple of 64
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    grad_ckpt: bool = False  # activation checkpointing per transformer block

    def to_dict(self):
        return asdict(self)


# Named sizes. 125m/350m are the main study; the larger ones exist so the
# max-model-size search can show how far FSDP pushes past DDP.
MODEL_CONFIGS: dict[str, dict] = {
    "tiny":     dict(n_layer=2,  n_head=2,  n_embd=64,   block_size=128, vocab_size=512),
    "gpt-125m": dict(n_layer=12, n_head=12, n_embd=768),
    "gpt-350m": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt-760m": dict(n_layer=24, n_head=16, n_embd=1536),
    "gpt-1.3b": dict(n_layer=24, n_head=32, n_embd=2048),
    "gpt-2.7b": dict(n_layer=32, n_head=32, n_embd=2560),
    "gpt-4b":   dict(n_layer=36, n_head=32, n_embd=3072),
    "gpt-6.7b": dict(n_layer=32, n_head=32, n_embd=4096),
    "gpt-13b":  dict(n_layer=40, n_head=40, n_embd=5120),
    "gpt-20b":  dict(n_layer=44, n_head=48, n_embd=6144),
}


def get_config(name: str, **overrides) -> GPTConfig:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_CONFIGS)}")
    cfg = dict(MODEL_CONFIGS[name])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return GPTConfig(**cfg)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # Uses FlashAttention / memory-efficient kernels when available.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)
        self.grad_ckpt = cfg.grad_ckpt

    def _forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

    def forward(self, x):
        if self.grad_ckpt and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying, as in GPT-2

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        """GPT-2 init. Called again after meta-device construction + sharding (FSDP)."""
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # scaled init for residual projections
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Analytic helpers (no model instance needed, so they work for sizes that
# would not fit in memory).
# ---------------------------------------------------------------------------

def num_params(cfg: GPTConfig, non_embedding: bool = False) -> int:
    d, L, V, T = cfg.n_embd, cfg.n_layer, cfg.vocab_size, cfg.block_size
    per_block = 12 * d * d + 2 * d  # qkv+proj+fc+proj weights, 2 LayerNorm weights
    if cfg.bias:
        per_block += 11 * d  # qkv 3d, proj d, fc 4d, proj d, 2 LayerNorm biases
    total = L * per_block + V * d + T * d + d * (2 if cfg.bias else 1)  # tied emb, pos emb, ln_f
    if non_embedding:
        total -= V * d + T * d
    return total


def flops_per_token(cfg: GPTConfig, seq_len: int) -> float:
    """Training FLOPs per token (fwd + bwd) following the PaLM appendix:
    6N for the dense matmuls plus 12*L*d*T for attention scores."""
    n = num_params(cfg, non_embedding=True) + cfg.vocab_size * cfg.n_embd  # lm_head matmul counts
    return 6 * n + 12 * cfg.n_layer * cfg.n_embd * seq_len

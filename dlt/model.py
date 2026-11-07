"""GPT-style decoder-only transformer (GPT-2 architecture, pre-LayerNorm).

Kept deliberately small and dependency-free so the same module is used by the
trainer on CPU (correctness tests) and GPU (benchmarks).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    vocab_size: int = 50304  # GPT-2 vocab padded to a multiple of 64
    seq_len: int = 1024
    dropout: float = 0.0
    bias: bool = True

    def to_dict(self):
        return asdict(self)

    # Analytical parameter count (matches GPT.num_params() exactly; tested).
    def num_params(self) -> int:
        d, L, V, T = self.d_model, self.n_layer, self.vocab_size, self.seq_len
        b = 1 if self.bias else 0
        per_block = (
            2 * (2 * d)                      # ln_1, ln_2 (weight + bias)
            + (d * 3 * d + b * 3 * d)        # qkv
            + (d * d + b * d)                # attn proj
            + (d * 4 * d + b * 4 * d)        # mlp fc
            + (4 * d * d + b * d)            # mlp proj
        )
        return V * d + T * d + L * per_block + 2 * d  # tok emb (tied) + pos emb + blocks + ln_f

    def block_params(self) -> int:
        return (self.num_params() - self.vocab_size * self.d_model
                - self.seq_len * self.d_model - 2 * self.d_model) // self.n_layer


# Named configurations. "tiny"/"mini" are for CPU validation; the GPT sizes
# follow GPT-2 / GPT-3 paper shapes and are used on GPUs.
CONFIGS: dict[str, GPTConfig] = {
    "tiny":     GPTConfig(n_layer=4,  n_head=4,  d_model=128,  vocab_size=256, seq_len=128),
    "mini":     GPTConfig(n_layer=6,  n_head=6,  d_model=384,  vocab_size=256, seq_len=256),
    "gpt-125m": GPTConfig(n_layer=12, n_head=12, d_model=768),
    "gpt-350m": GPTConfig(n_layer=24, n_head=16, d_model=1024),
    "gpt-760m": GPTConfig(n_layer=24, n_head=16, d_model=1536),
    "gpt-1.3b": GPTConfig(n_layer=24, n_head=32, d_model=2048),
    "gpt-2.7b": GPTConfig(n_layer=32, n_head=32, d_model=2560),
    "gpt-6.7b": GPTConfig(n_layer=32, n_head=32, d_model=4096),
    "gpt-13b":  GPTConfig(n_layer=40, n_head=40, d_model=5120),
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying (GPT-2)
        self.activation_checkpointing = False
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        self.apply(self._init_weights)
        # GPT-2 scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layer))
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.reset_parameters()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            if self.activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss

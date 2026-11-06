"""Data loading.

Batches are drawn deterministically from (seed, step): every rank builds the
same *global* batch and keeps only its own slice. That makes a run on N GPUs
with micro-batch B mathematically identical to a 1-GPU run with micro-batch
N*B, which is what the equivalence test relies on.
"""
from __future__ import annotations

import numpy as np
import torch


class TokenBatcher:
    def __init__(
        self,
        path: str | None,
        seq_len: int,
        micro_batch: int,
        rank: int,
        world_size: int,
        vocab_size: int,
        seed: int = 1337,
        device: str | torch.device = "cpu",
    ):
        self.seq_len = seq_len
        self.micro_batch = micro_batch
        self.rank = rank
        self.world_size = world_size
        self.vocab_size = vocab_size
        self.seed = seed
        self.device = torch.device(device)
        self.tokens = None
        if path is not None:
            self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
            if len(self.tokens) <= seq_len + 1:
                raise ValueError(f"{path} has only {len(self.tokens)} tokens")

    @property
    def synthetic(self) -> bool:
        return self.tokens is None

    def get(self, step: int, micro_step: int = 0):
        g = np.random.default_rng([self.seed, step, micro_step])
        global_b = self.micro_batch * self.world_size
        lo = self.rank * self.micro_batch
        hi = lo + self.micro_batch
        if self.synthetic:
            # Random tokens: fine for throughput/memory benchmarks, where the
            # values of the tokens do not affect speed.
            buf = g.integers(0, self.vocab_size, size=(global_b, self.seq_len + 1))[lo:hi]
            buf = torch.from_numpy(buf.astype(np.int64))
        else:
            starts = g.integers(0, len(self.tokens) - self.seq_len - 1, size=global_b)[lo:hi]
            buf = torch.from_numpy(
                np.stack([self.tokens[s : s + self.seq_len + 1].astype(np.int64) for s in starts])
            )
        if self.device.type == "cuda":
            buf = buf.pin_memory().to(self.device, non_blocking=True)
        else:
            buf = buf.to(self.device)
        return buf[:, :-1], buf[:, 1:]

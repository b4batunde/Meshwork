"""Token data sources.

The sampler is the key piece for distributed correctness: at step s every rank
derives the SAME global batch from (seed, s) and takes its own contiguous
slice. A single-process run with the full global batch therefore sees exactly
the same tokens as an N-rank run, which is what lets the tests assert that
DDP/FSDP losses match the single-device baseline.

Sources:
  synthetic  uniform random tokens (throughput benchmarking; any vocab)
  stdlib     byte-level tokens from Python's own stdlib source (real text,
             no download needed; vocab 256)
  path.bin   uint16 token file (e.g. produced by scripts/prepare_openwebtext.py)
"""
from __future__ import annotations

import glob
import os
import sysconfig

import numpy as np
import torch


def _stdlib_corpus(max_bytes: int = 32_000_000) -> np.ndarray:
    cache = os.path.join(os.path.dirname(__file__), "..", "data", "stdlib_bytes.npy")
    cache = os.path.abspath(cache)
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    root = sysconfig.get_paths()["stdlib"]
    chunks, total = [], 0
    for path in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        try:
            with open(path, "rb") as f:
                b = f.read()
        except OSError:
            continue
        chunks.append(b)
        total += len(b)
        if total >= max_bytes:
            break
    arr = np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, arr)
    return arr


class TokenSource:
    def __init__(self, kind: str, vocab_size: int, seq_len: int, seed: int = 0):
        self.kind, self.vocab_size, self.seq_len, self.seed = kind, vocab_size, seq_len, seed
        if kind == "synthetic":
            self.tokens = None
        elif kind == "stdlib":
            assert vocab_size >= 256, "stdlib source is byte-level; need vocab >= 256"
            self.tokens = _stdlib_corpus()
        else:
            self.tokens = np.memmap(kind, dtype=np.uint16, mode="r")
        if self.tokens is not None:
            n = len(self.tokens)
            split = int(n * 0.95)
            self.train_range, self.val_range = (0, split), (split, n)

    def global_batch(self, step: int, global_bs: int, split: str = "train"):
        """Deterministic global batch for a step: (x, y) of shape [global_bs, seq_len]."""
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + step * 7 + (0 if split == "train" else 1))
        T = self.seq_len
        if self.tokens is None:
            data = torch.randint(0, self.vocab_size, (global_bs, T + 1), generator=g)
        else:
            lo, hi = self.train_range if split == "train" else self.val_range
            starts = torch.randint(lo, hi - T - 1, (global_bs,), generator=g).tolist()
            data = torch.stack([torch.from_numpy(np.asarray(self.tokens[s:s + T + 1], dtype=np.int64))
                                for s in starts])
        return data[:, :-1].contiguous(), data[:, 1:].contiguous()

    def local_batch(self, step: int, micro_bs: int, rank: int, world: int, split: str = "train"):
        x, y = self.global_batch(step, micro_bs * world, split)
        sl = slice(rank * micro_bs, (rank + 1) * micro_bs)
        return x[sl], y[sl]

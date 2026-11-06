# Measured on CPU

**Measured (CPU, gloo backend, real train.py runs).** These check correctness and memory layout; they say nothing about GPU speed.

Same global batch of 8 sequences, 30 optimizer steps, fp32 (`tiny` model):

| Run | Max relative loss difference vs single process | Loss, step 0 -> 29 |
|---|---|---|
| DDP x2 | 1.1e-07 | 6.2546 -> 6.2379 |
| FSDP x2 | 1.1e-07 | 6.2546 -> 6.2379 |

Per-rank model state (params + grads + AdamW m, v), read from the live tensors:

| Model | Strategy | Processes | Measured | Simulated | Difference | Share of DDP's |
|---|---|---|---|---|---|---|
| gpt-125m | DDP | 1 | 1,897.8 MiB | 1,897.8 MiB | +0.00% | 100.0% |
| gpt-125m | FSDP | 1 | 1,897.8 MiB | 1,897.8 MiB | +0.00% | 100.0% |
| gpt-125m | FSDP | 2 | 948.9 MiB | 948.9 MiB | +0.00% | 50.0% |
| tiny | DDP | 1 | 2,181.0 KiB | 2,181.0 KiB | +0.00% | 100.0% |
| tiny | FSDP | 2 | 1,090.5 KiB | 1,090.5 KiB | +0.00% | 50.0% |
| tiny | FSDP | 4 | 545.2 KiB | 545.2 KiB | +0.00% | 25.0% |

FSDP's per-rank share is exactly 1/N here because every dimension divides evenly (no shard padding), and the simulator's model-state arithmetic matches the live tensors byte for byte.

#!/usr/bin/env bash
# Free MEASURED results on Kaggle's 2x T4 notebooks (if that accelerator option is
# offered: Settings -> Accelerator -> "GPU T4 x2"). In a notebook cell:
#
#   !git clone https://github.com/<you>/distributed-llm-training && cd distributed-llm-training \
#       && pip install -q tiktoken && bash scripts/kaggle_2xT4.sh
#
# T4s have no bf16 and no NVLink: runs use fp16 with loss scaling, PCIe Gen3 between GPUs.
# Takes ~1 to 1.5 hours. Results land in results/ and the README's Measured block, and
# the last step prints Simulated vs Measured for the same configurations.
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi
nvidia-smi topo -m | tee results/topology_t4.txt
python -m pytest -q tests/

COMMON="--dtype fp16 --gpu-type T4"
MODELS="gpt-125m" GPUS="1 2" MB=4 GLOBAL_MB=16 STEPS=30 NOP2P=0 EXTRA="$COMMON" bash bench/run_scaling.sh
MODELS="gpt-125m" NPROC=2 BATCHES="1 2 4 8" EXTRA="$COMMON" bash bench/memory_sweep.sh

python bench/analyze.py --update-readme
python bench/simulate.py --compare results/scaling.jsonl --link pcie3-host | tee results/sim_vs_measured_t4.md

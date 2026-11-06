#!/bin/bash
# SLURM job for a single 4-GPU node (written for ASU Sol; adjust partition/QOS/GPU
# type for other clusters, and check `sinfo` / the Sol docs for current names).
#
#   sbatch slurm/sol_scaling.sbatch
#   GPU=h100 sbatch -G h100:4 slurm/sol_scaling.sbatch
#
#SBATCH --job-name=dlt-scaling
#SBATCH -p general
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G a100:4
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 0-06:00:00
#SBATCH -o results/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load mamba/latest 2>/dev/null || true
source activate dlt 2>/dev/null || true   # env created per README

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
nvidia-smi
nvidia-smi topo -m | tee results/topology.txt   # NVLink vs PCIe matters a lot for the analysis

python -m pytest -q tests/ || { echo "correctness tests failed"; exit 1; }

GPU_TYPE=${GPU:-A100}
EXTRA="--gpu-type ${GPU_TYPE^^}" bash bench/run_scaling.sh
EXTRA="--gpu-type ${GPU_TYPE^^}" bash bench/memory_sweep.sh
python bench/find_max_model.py --nproc 4
python bench/analyze.py --update-readme   # writes measured numbers + plots into README.md

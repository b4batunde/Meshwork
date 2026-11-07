#!/usr/bin/env bash
# Measured GPU study. Run on one node with >= 4 GPUs (e.g. a 4xA100 cloud instance).
# Produces results/gpu/*.json, summary.json and figures/*.png.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/gpu
mkdir -p "$OUT"
MB=${MB:-8}; STEPS=${STEPS:-60}; WARM=${WARM:-10}
GPU=${GPU:-H100-80GB}   # H100-80GB | A100-80GB (sets bf16 peak FLOPs for MFU)
case "$GPU" in H100*) PEAK=989 ;; A100*) PEAK=312 ;; *) PEAK=${PEAK:?set PEAK to the GPU bf16 TFLOPS} ;; esac
COMMON="--data synthetic --micro-batch $MB --steps $STEPS --warmup-steps $WARM --quiet"

nvidia-smi --query-gpu=name,memory.total --format=csv > "$OUT/gpus.csv"
nvidia-smi topo -m > "$OUT/topology.txt" || true

for W in 2 4; do
  torchrun --standalone --nproc_per_node "$W" -m dlt.commbench --out "$OUT/commbench_w$W.json"
done

for M in gpt-125m gpt-350m; do
  python -m dlt.train --strategy single --model "$M" $COMMON --out "$OUT/${M}_single_w1.json"
  for S in ddp fsdp; do
    for W in 1 2 4; do
      torchrun --standalone --nproc_per_node "$W" -m dlt.train --strategy "$S" --model "$M" \
        $COMMON --comm-probe-steps 10 --out "$OUT/${M}_${S}_w${W}.json"
    done
  done
done

# Largest model that fits: walk up the ladder until OOM (act. checkpointing on).
for S in ddp fsdp; do
  for M in gpt-760m gpt-1.3b gpt-2.7b gpt-6.7b gpt-13b; do
    EXTRA=""; [ "$S" = fsdp ] && EXTRA="--meta-init"
    # timeout: if only one rank OOMs the others can hang in a collective
    timeout 900 torchrun --standalone --nproc_per_node 4 -m dlt.train --strategy "$S" --model "$M" \
      --data synthetic --micro-batch "$MB" --steps 4 --warmup-steps 1 --act-ckpt $EXTRA --quiet \
      --out "$OUT/maxprobe_${S}_${M}.json" || true
  done
done

python scripts/summarize_measured.py --dir "$OUT" --peak-tflops "$PEAK"
python scripts/plot_measured.py --dir "$OUT"

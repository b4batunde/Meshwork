#!/usr/bin/env bash
# Peak memory vs micro-batch for DDP and FSDP, with and without activation
# checkpointing. Shows where activation memory overtakes the (sharded) model
# state, i.e. where FSDP's theoretical savings stop being the whole story.
#
#   bash bench/memory_sweep.sh
#   MODELS="gpt-350m" NPROC=4 BATCHES="1 2 4 8 16 32" bash bench/memory_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"gpt-125m gpt-350m"}
NPROC=${NPROC:-4}
BATCHES=${BATCHES:-"1 2 4 8 16 32"}
OUT=${OUT:-results/memsweep.jsonl}
EXTRA=${EXTRA:-}

for model in $MODELS; do
  for strat in ddp fsdp; do
    for ckpt in "" "--grad-ckpt"; do
      for mb in $BATCHES; do
        echo ">>> $model $strat x$NPROC mb=$mb ${ckpt}"
        torchrun --standalone --nproc_per_node "$NPROC" train.py --model "$model" --strategy "$strat" \
          --micro-batch "$mb" --steps 6 --timing-warmup 2 --tag memsweep --out "$OUT" $ckpt $EXTRA
        [[ $? -eq 3 ]] && { echo "    OOM, stopping this series"; break; }
      done
    done
  done
done
echo "done -> $OUT"

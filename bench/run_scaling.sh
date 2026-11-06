#!/usr/bin/env bash
# Full scaling sweep: {single, DDP, FSDP} x {1, 2, 4 GPUs} x models.
#
#   bash bench/run_scaling.sh                       # gpt-125m and gpt-350m
#   MODELS="gpt-125m" GPUS="1 2" bash bench/run_scaling.sh
#
# Env knobs (defaults in brackets):
#   MODELS [gpt-125m gpt-350m]  GPUS [1 2 4]  MB [8]  STEPS [40]
#   GLOBAL_MB [32]  sequences per step for the strong-scaling runs
#   DATA []         token .bin file; empty = synthetic tokens (fine for speed/memory)
#   OUT [results/scaling.jsonl]   EXTRA []  extra flags for train.py (e.g. "--compile")
#   PROFILE [1]     also run a torch.profiler pass to split comm vs compute kernel time
#   NOP2P [1]       repeat the largest-N weak runs with NCCL_P2P_DISABLE=1 (no NVLink/P2P:
#                   traffic goes through host memory, approximating a PCIe-only box)
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"gpt-125m gpt-350m"}
GPUS=${GPUS:-"1 2 4"}
MB=${MB:-8}
STEPS=${STEPS:-40}
GLOBAL_MB=${GLOBAL_MB:-32}
OUT=${OUT:-results/scaling.jsonl}
DATA_FLAG=${DATA:+--data $DATA}
EXTRA=${EXTRA:-}
PROFILE=${PROFILE:-1}
NOP2P=${NOP2P:-1}
MAXN=$(echo $GPUS | tr ' ' '\n' | sort -n | tail -1)

launch() {  # launch <nproc> <args...>
  local n=$1; shift
  echo ">>> n=$n $*"
  if [[ $n -eq 1 && " $* " == *" --strategy single "* ]]; then
    python train.py "$@" || echo "!!! run failed (exit $?)"
  else
    torchrun --standalone --nproc_per_node "$n" train.py "$@" || echo "!!! run failed (exit $?)"
  fi
}

for model in $MODELS; do
  base=(--model "$model" --steps "$STEPS" --out "$OUT" $DATA_FLAG $EXTRA)

  # Baseline
  launch 1 "${base[@]}" --strategy single --micro-batch "$MB" --tag weak
  launch 1 "${base[@]}" --strategy single --micro-batch "$GLOBAL_MB" --tag strong \
    || true  # may OOM at large global batch; analysis falls back gracefully

  for n in $GPUS; do
    for strat in ddp fsdp; do
      # Weak scaling: fixed per-GPU batch, global batch grows with N.
      launch "$n" "${base[@]}" --strategy "$strat" --micro-batch "$MB" --tag weak
      # Strong scaling: fixed global batch, split across N GPUs.
      if (( GLOBAL_MB % n == 0 )); then
        launch "$n" "${base[@]}" --strategy "$strat" --micro-batch $((GLOBAL_MB / n)) --tag strong
      fi
      if (( n > 1 )); then
        # Same run with gradient sync disabled -> compute-only step time.
        # (full step time - compute-only) = exposed (non-overlapped) communication.
        launch "$n" "${base[@]}" --strategy "$strat" --micro-batch "$MB" --tag compute_only --no-grad-sync
        if [[ $PROFILE == 1 ]]; then
          launch "$n" "${base[@]}" --strategy "$strat" --micro-batch "$MB" --tag profile --profile --steps 15
        fi
        if [[ $NOP2P == 1 && $n -eq $MAXN ]]; then
          NCCL_P2P_DISABLE=1 launch "$n" "${base[@]}" --strategy "$strat" --micro-batch "$MB" --tag weak_nop2p
        fi
        if [[ $strat == ddp && $n -eq $MAXN ]]; then
          # Tests the simulated tied-embedding bottleneck: halve gradient bytes.
          launch "$n" "${base[@]}" --strategy ddp --micro-batch "$MB" --tag weak_bf16comm --ddp-bf16-comm
          if [[ $NOP2P == 1 ]]; then
            NCCL_P2P_DISABLE=1 launch "$n" "${base[@]}" --strategy ddp --micro-batch "$MB" --tag weak_nop2p_bf16comm --ddp-bf16-comm
          fi
        fi
      fi
    done
  done
done
echo "done -> $OUT"

#!/usr/bin/env bash
# Multi-seed extension of run_reference_protocol.sh.
#
# Seed 145 (their configs/default.yaml default) was run first and gave test-selected MAD 4.0885.
# One seed establishes that the selection effect exists; it cannot say how much of the specific
# 4.09 is seed luck. This runs the remaining seeds under the identical unmodified protocol so the
# selection effect can be reported as a mean +/- sd over seeds, the same convention the paper uses
# for its own three-seed numbers.
#
# Nothing in Transformer_Rehabilitation/ is edited: seeds go through their own --seed flag.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../Transformer_Rehabilitation" || exit 1

PY="${PY:-$HERE/../.venv/Scripts/python.exe}"
EPOCHS="${EPOCHS:-2000}"
SEEDS="${SEEDS:-146 147}"

for SEED in $SEEDS; do
  OUT="./repro_s${SEED}"
  if [ -f "$OUT/eval.log" ]; then
    echo "=== seed $SEED already done, skipping"
    continue
  fi
  mkdir -p "$OUT"
  echo "=== reference protocol: ex=Kimore_ex1 seed=$SEED epochs=$EPOCHS  ($(date))"
  "$PY" train.py --config configs/default.yaml --ex Kimore_ex1 \
    --epoch "$EPOCHS" --seed "$SEED" --save_dir "$OUT" \
    > "$OUT/train.log" 2>&1 || { echo "TRAIN FAILED (seed $SEED)"; exit 1; }

  "$PY" eval.py --config configs/default.yaml --ex Kimore_ex1 \
    --checkpoint "$OUT/Kimore_ex1.pt" > "$OUT/eval.log" 2>&1 || { echo "EVAL FAILED (seed $SEED)"; exit 1; }

  echo "--- seed $SEED done ($(date))"
  tail -6 "$OUT/eval.log"
done

echo "REFERENCE_REPRO_SEEDS_DONE"

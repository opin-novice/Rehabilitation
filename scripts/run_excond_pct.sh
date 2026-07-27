#!/usr/bin/env bash
# Fairness arm: PCT with the head conditioned on a one-hot exercise id.
#
# Why: PointCloudTransformerRegressor ACCEPTED num_exercises and silently ignored it, so in the
# pooled comparison the baseline was scored blind to which exercise it was looking at while EGRU
# one-hots the exercise id and the mean-predictor floor is per-exercise. --exercise-cond is the
# fixed arm; the flag is opt-in so the banked runs stay bit-for-bit reproducible.
#
# Writes to a SEPARATE out dir: the checkpoint tag has no exercise-cond marker, so training here
# with --out outputs/cde_block2 would clobber the banked pct_pooled*_s*_f*.pt Block-3 baselines.
#
# Budget matches the banked arms exactly (60 epochs, the train_baseline_pct.py default) so the
# only variable that moves between banked and this is the conditioning.
set -u
O=outputs/cde_block2_excond
mkdir -p "$O"

for S in 0 1 2; do
  for ARM in "" "--aug-rot"; do
    TAG=$([ -z "$ARM" ] && echo clean || echo rot)
    LOG="$O/tr_pct_${TAG}_s${S}.log"
    CK="$O/pct_pooled$([ -z "$ARM" ] && echo "" || echo rot)_s${S}_f4.pt"
    if [ -f "$CK" ]; then
      echo "skip seed $S $TAG (have $CK)"
      continue
    fi
    echo "=== seed $S  arm $TAG -> $LOG"
    python src/train_baseline_pct.py --cv --pooled --exercise-cond $ARM \
      --seed "$S" --out "$O" > "$LOG" 2>&1 || { echo "FAILED seed $S $TAG"; exit 1; }
  done
done
echo "EXCOND_TRAINING_DONE"

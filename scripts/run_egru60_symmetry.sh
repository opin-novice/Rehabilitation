#!/usr/bin/env bash
# Budget-symmetry check: EGRU trained at the PCT baseline's epoch budget.
#
# Why: train_egru.py defaults to 80 epochs, train_baseline_pct.py to 60. That asymmetry runs in
# OUR favour, and it is the mirror image of the fairness holes we closed on the baseline
# (exercise conditioning, best-case resampling operator). A reviewer is entitled to ask whether
# the clean-accuracy TIE survives once both models get the same budget.
#
# Pass condition is the TIE, not a win: the paper claims EGRU (6.73+/-0.30) and PCT (6.47+/-0.20)
# are indistinguishable inside the 0.33 MAD nondeterminism floor. The claim breaks if EGRU@60
# falls more than 0.33 behind PCT, NOT if it merely fails to beat it.
#
# Both arms of Table 1 are run: chiral SO(3) is the adopted model, O(3) is the parity-even control.
set -u
O=outputs/cde_block2_egru60
mkdir -p "$O"

for S in 0 1 2; do
  for ARM in "--chiral" ""; do
    TAG=$([ -n "$ARM" ] && echo chi || echo o3)
    CK="$O/$([ -n "$ARM" ] && echo egruchi || echo egru)_s${S}_pooled_f4.pt"
    if [ -f "$CK" ]; then
      echo "skip seed $S $TAG (have $CK)"
      continue
    fi
    echo "=== seed $S  arm $TAG"
    python src/train_egru.py --cv $ARM --epochs 60 --seed "$S" --out "$O" \
      > "$O/tr_egru60_${TAG}_s${S}.log" 2>&1 || { echo "FAILED seed $S $TAG"; exit 1; }
  done
done
echo "EGRU60_TRAINING_DONE"

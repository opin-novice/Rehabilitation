#!/usr/bin/env bash
# FULL steelman of the PCT baseline: both fairness fixes applied together.
#
#   --exercise-cond : one-hot exercise id on the head, as EGRU and the per-exercise
#                     mean-predictor floor already get. Tested alone -> null.
#   --k 20          : the neighbourhood size the REFERENCE implementation actually uses
#                     (Transformer_Rehabilitation/core/models/curvenet_cls.py:19, "# k=20").
#                     Our CurveNetEncoder already defaults to k=20; train_baseline_pct.py's
#                     --k default of 10 shadowed it, so every banked PCT run used half the
#                     reference's neighbourhood. This is the larger of the two fixes and the
#                     one most likely to move a number.
#
# These are run TOGETHER because that is the strongest form of the baseline, and a reviewer
# asking "did you give it a fair shot" means the conjunction, not either fix alone.
#
# Separate out dir: the checkpoint tag encodes neither k nor conditioning, so training into
# cde_block2 would clobber the banked Block-3 baselines.
set -u
O=outputs/cde_block2_steelman
mkdir -p "$O"

for S in 0 1 2; do
  for ARM in "" "--aug-rot"; do
    TAG=$([ -z "$ARM" ] && echo clean || echo rot)
    CK="$O/pct_pooled$([ -z "$ARM" ] && echo "" || echo rot)_s${S}_f4.pt"
    if [ -f "$CK" ]; then
      echo "skip seed $S $TAG (have $CK)"
      continue
    fi
    echo "=== seed $S  arm $TAG"
    python src/train_baseline_pct.py --cv --pooled --exercise-cond --k 20 $ARM \
      --seed "$S" --out "$O" > "$O/tr_pct_${TAG}_s${S}.log" 2>&1 \
      || { echo "FAILED seed $S $TAG"; exit 1; }
  done
done
echo "STEELMAN_TRAINING_DONE"

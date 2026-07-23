#!/usr/bin/env bash
# Repair seeds 1,2 after the fold-partition bug: retrain invgru on the correct per-seed folds,
# then re-run every seed-1/2 sweep (egru/pct checkpoints were fine; only their eval used the
# wrong partition). Idempotent via result-file markers.
set -u
cd /d/Rehabilation
export PYTHONUNBUFFERED=1
O=outputs/cde_block2
run() { local m=$1 l=$2; shift 2; [[ -f $m ]] && { echo "SKIP $m"; return; }
        echo ">>> $*"; if "$@" >"$l" 2>&1; then echo "    OK $m"; else echo "    FAIL $l"; fi; }

for S in 1 2; do
  echo "==================== SEED $S ===================="
  run $O/invariant_controls_s${S}.json    outputs/tr_invgru_s$S.log \
      python src/invariant_controls.py --cv --seed $S
  run $O/block3_s${S}_results.json        outputs/sw_b3_egru_s$S.log \
      python src/block23_experiments.py --block 3 --cv --method egru   --model-seed $S
  run $O/block3_invgru_s${S}_results.json outputs/sw_b3_invgru_s$S.log \
      python src/block23_experiments.py --block 3 --cv --method invgru --model-seed $S
  run $O/block3_rot_s${S}_results.json    outputs/sw_b3_pctrot_s$S.log \
      python src/block23_experiments.py --block 3 --cv --method egru   --pct-rot --model-seed $S
  run $O/block2_aug_s${S}_results.json    outputs/sw_b2_egru_s$S.log \
      python src/block23_experiments.py --block 2 --cv --method egru   --aug --model-seed $S
done
echo "==================== SEED 1/2 REPAIR COMPLETE ===================="
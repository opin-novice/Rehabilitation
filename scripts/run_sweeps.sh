#!/usr/bin/env bash
# All Block-2 and Block-3 sweeps, across 3 seeds, idempotent (skips if results json exists).
# Result files land as block{2,3}{sfx}_s{seed}_results.json per block23_experiments naming.
set -u
cd /d/Rehabilation
export PYTHONUNBUFFERED=1
O=outputs/cde_block2

run() {  # run <marker> <log> <command...>
  local marker=$1 log=$2; shift 2
  if [[ -f $marker ]]; then echo "SKIP (done) $marker"; return; fi
  echo ">>> $* -> $log"
  if "$@" > "$log" 2>&1; then echo "    OK $marker"; else echo "    FAIL $log"; fi
}

for S in 0 1 2; do
  echo "==================== SEED $S ===================="
  # --- BLOCK 3 (cross-viewpoint): ours=egru, invgru; baseline pct clean + pct+rot fairness arm ---
  run $O/block3_egru_s${S}_results.json      outputs/sw_b3_egru_s$S.log     \
      python src/block23_experiments.py --block 3 --cv --method egru   --model-seed $S
  run $O/block3_invgru_s${S}_results.json    outputs/sw_b3_invgru_s$S.log   \
      python src/block23_experiments.py --block 3 --cv --method invgru --model-seed $S
  run $O/block3_rot_s${S}_results.json       outputs/sw_b3_pctrot_s$S.log   \
      python src/block23_experiments.py --block 3 --cv --method egru   --pct-rot --model-seed $S
  # --- BLOCK 2 (irregular sampling): augmented arms, ours=egruaug vs pct+dropaug ---
  # invgru is NOT swept here: it was never drop-augmented, so it would collapse like the
  # un-augmented egru did and produce a misleading row. The R-as-denoiser finding is carried
  # by the egru arm, which HAS the matched augmentation budget.
  run $O/block2_aug_egru_s${S}_results.json  outputs/sw_b2_egru_s$S.log     \
      python src/block23_experiments.py --block 2 --cv --method egru   --aug --model-seed $S
done
echo "==================== SWEEPS COMPLETE ===================="
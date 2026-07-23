#!/usr/bin/env bash
# ==============================================================================
# CHIRALITY CAMPAIGN -- close the parity hole (reviewer Critique 3).
#
# Trains the SO(3) (parity-odd) variants of BOTH headline arms and re-runs the
# Block-3 viewpoint sweep on them. Both arms are upgraded together: giving only
# the EGRU the new pseudo-scalars would make the co-headline comparison bogus.
#
# The gate: Block-3 degradation MUST stay 0.000 at every angle. Rotations have
# det = +1, so pseudo-scalars are exactly rotation-invariant (certify_egru E8).
# If it moves, the fix broke the theorem and we revert.
#
# Idempotent via results-file markers -- survives a power cut (it has before).
# ==============================================================================
set -u
cd /d/Rehabilation
export PYTHONUNBUFFERED=1
O=outputs/cde_block2
mkdir -p outputs

run() { local m=$1 l=$2; shift 2; [[ -f $m ]] && { echo "SKIP $m"; return; }
        echo ">>> $*"; if "$@" >"$l" 2>&1; then echo "    OK $m"; else echo "    FAIL -> $l"; fi; }

for S in 0 1 2; do
  echo "==================== CHIRAL SEED $S ===================="

  # 1. hand-crafted control, chiral (fast) -- the co-headline must stay matched
  run $O/invariant_controls_invgruchi_s${S}.json  outputs/tr_invgruchi_s$S.log \
      python src/invariant_controls.py --cv --chiral --seed $S

  # 2. the proposed model, chiral
  run $O/egruchi_s${S}_results.json               outputs/tr_egruchi_s$S.log \
      python src/train_egru.py --cv --chiral --seed $S

  # 3. Block-3 viewpoint sweep on both chiral arms -- degradation must remain 0.000
  run $O/block3_chi_s${S}_results.json            outputs/sw_b3_egruchi_s$S.log \
      python src/block23_experiments.py --block 3 --cv --method egru   --chiral --model-seed $S
  run $O/block3_invgru_chi_s${S}_results.json     outputs/sw_b3_invgruchi_s$S.log \
      python src/block23_experiments.py --block 3 --cv --method invgru --chiral --model-seed $S
done
echo "==================== CHIRAL CAMPAIGN COMPLETE ===================="

#!/usr/bin/env bash
# Idempotent resume of the 3-seed campaign. Each arm is guarded by its results-file marker:
# if the marker exists the arm is skipped, so re-running after a crash costs nothing and
# resumes exactly where it stopped. (A partial --cv run leaves stale fold checkpoints but no
# results file, so it is correctly re-run in full and overwrites them.)
set -u
cd /d/Rehabilation
export PYTHONUNBUFFERED=1
O=outputs/cde_block2

run() {  # run <marker-file> <logfile> <command...>
  local marker=$1 log=$2; shift 2
  if [[ -f $marker ]]; then echo "SKIP  (done) $marker"; return; fi
  echo ">>> RUN  $* -> $log"
  if "$@" > "$log" 2>&1; then echo "    OK  $marker"; else echo "    FAIL $log"; fi
}

for S in 1 2; do
  echo "==================== SEED $S ===================="
  run $O/egru_s${S}_results.json            outputs/tr_egru_s$S.log      python src/train_egru.py --cv --seed $S
  run $O/invariant_controls_s${S}.json      outputs/tr_invgru_s$S.log    python src/invariant_controls.py --cv --seed $S
  run $O/pct_results_ex1_s${S}.json         outputs/tr_pct_s$S.log       python src/train_baseline_pct.py --cv --pooled --seed $S
  run $O/pct_results_ex1_rot_s${S}.json     outputs/tr_pctrot_s$S.log    python src/train_baseline_pct.py --cv --pooled --aug-rot --seed $S
  run $O/egruaug_s${S}_results.json         outputs/tr_egruaug_s$S.log   python src/train_egru.py --cv --aug-drop 0.3 --seed $S
  run $O/pct_results_ex1_aug_s${S}.json     outputs/tr_pctaug_s$S.log    python src/train_baseline_pct.py --cv --pooled --aug-drop 0.3 --seed $S
done
echo "==================== RESUME COMPLETE ===================="
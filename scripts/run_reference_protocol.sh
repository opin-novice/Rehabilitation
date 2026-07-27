#!/usr/bin/env bash
# Task 3: reproduce the reference paper's OWN protocol by running its OWN code, unmodified.
#
# Nothing in Transformer_Rehabilitation/ is edited. Everything below is set through the CLI flags
# their config.py already exposes (--ex, --epoch, --seed, --save_dir), so the run is theirs.
#
# What their protocol does, verified by reading the source:
#   engine/trainer.py:59-65  selects the saved checkpoint by minimum TEST MAD. train.py builds only
#                            (train_loader, test_loader) -- there is no validation split anywhere.
#   engine/trainer.py:73     evaluate_mad does NOT inverse-transform, so the per-epoch "val MAD="
#                            printed to stdout is in STANDARDISED units (StandardScaler, sigma=8.466).
#   engine/evaluator.py      evaluate() DOES inverse-transform, so eval.py reports SCORE units.
#                            The two differ by 8.466x; conflating them inflates any comparison.
#
# We capture the full per-epoch curve to the log, which lets us recover from one unmodified run:
#   - their reported statistic : min over epochs (what the checkpoint selection bakes in)
#   - the honest statistic     : final-epoch test MAD
#   - the inflation            : the difference between the two
#
# Their released checkpoint already scores 5.3461 score units / 0.6315 standardised via their own
# eval.py, and our single-exercise test-selected number is 5.21+/-0.19 -- i.e. we already reproduce
# them under matched protocol. This run answers the one residual question: does their recipe, at the
# full 2000-epoch budget, reach the published 0.185?
set -u
cd "$(dirname "$0")/../Transformer_Rehabilitation" || exit 1

SEED=${1:-145}          # their configs/default.yaml seed
EPOCHS=${2:-2000}       # their configs/default.yaml epoch budget
OUT=./repro_s${SEED}
mkdir -p "$OUT"

echo "=== reference protocol: ex=Kimore_ex1 seed=$SEED epochs=$EPOCHS"
python train.py --config configs/default.yaml --ex Kimore_ex1 \
  --epoch "$EPOCHS" --seed "$SEED" --save_dir "$OUT" \
  > "$OUT/train.log" 2>&1 || { echo "TRAIN FAILED"; exit 1; }

# Their eval.py on the checkpoint their own selection rule chose.
python eval.py --config configs/default.yaml --ex Kimore_ex1 \
  --checkpoint "$OUT/Kimore_ex1.pt" > "$OUT/eval.log" 2>&1 || { echo "EVAL FAILED"; exit 1; }

echo "REFERENCE_REPRO_DONE"
tail -6 "$OUT/eval.log"

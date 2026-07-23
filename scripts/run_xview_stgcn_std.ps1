# run_xview_stgcn_std.ps1 -- THE CONTROL ANCHOR (Step 1)
#
# Trains ST-GCN under the PUBLISHED configuration to test whether our NTU pipeline can reproduce
# the ~85-88% X-View benchmark. If it cannot, every NTU number in the paper is uninterpretable and
# the "EGRU beats ST-GCN" comparison is void.
#
# What differs from the earlier (64.91%) run -- all three were confounds, not one:
#   1. ARCHITECTURE  stgcn_full = published ST-GCN (K=3 spatial-configuration partitioning,
#                    10 blocks, learnable edge importance, 3.10M params) instead of the compact
#                    single-adjacency 6-block 1.74M variant that cannot reach 88 on any recipe.
#   2. PREPROCESSING --view-norm  = PreNormalize3D camera canonicalisation (published ST-GCN
#                    depends on this for X-View) instead of per-frame root-relative, which removes
#                    orientation not at all and destroys the motion trajectory.
#                    --sample uniform = spread 100 frames over the whole action instead of
#                    truncating to the causal first 100.
#   3. RECIPE        SGD + nesterov, momentum 0.9, wd 5e-4, cosine + 5-epoch linear warmup,
#                    80 epochs (pyskl's ST-GCN recipe) instead of AdamW/60ep.
#                    lr 0.05 = pyskl's 0.1 linear-scaled from their batch 128 to our batch 64.
#
# RESUMABLE: re-run this exact command after any interruption; it continues from the last
# COMPLETED epoch. Do not run two copies at once (they fight over the resume file).
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
python src/run_xview_evaluation.py --data data/ntu60_3danno.pkl --models stgcn_full `
  --view-norm --sample uniform `
  --optimizer sgd --lr 0.05 --wd 5e-4 --momentum 0.9 --warmup-epochs 5 `
  --epochs 80 --batch-size 64 --val-frac 0.05 --seed 42 `
  --out outputs/ntu_stgcn_std

# run_xview_stgcn_rot.ps1 -- THE CONTROL ANCHOR, attempt 2 (full pyskl parity)
#
# Adds the two remaining pyskl components to the 80.38% run, which was itself +15.5 over the
# original 64.91% (architecture + view-norm + SGD recipe):
#
#   --sample cyclic   pyskl UniformSample. Wraps short clips instead of zero-padding them.
#                     75% of NTU clips are <100 frames, and ST-GCN global-average-pools over the
#                     WHOLE T (it ignores `length`), so three quarters of the dataset had its
#                     pooled feature diluted by padding -- which BatchNorm turns into a non-zero
#                     constant rather than a harmless zero. Measured: padding fraction 0.424 -> 0.040
#                     (and the 0.040 residual is just the root joint zeroed by root-relative).
#                     Also gives temporal augmentation + real clip diversity.
#
#   --rot-aug 0.3     pyskl RandomRot, train split ONLY. theta=0.3 rad ~ +/-17 deg per axis.
#                     This is a MILD regulariser absorbing residual PreNormalize3D error. It is NOT
#                     "train the baseline on all viewpoints" and does NOT grant rotation invariance
#                     at the 45/90/180 deg scale the viewpoint theorem addresses. The adversarial
#                     control is theta=pi -- a SEPARATE experiment. Do not conflate them.
#
# Everything else matches the 80.38% run: published ST-GCN (3.10M, K=3 partitioning, 10 blocks,
# edge importance), PreNormalize3D view canonicalisation, SGD+nesterov lr0.05 wd5e-4 mom0.9,
# cosine + 5-epoch warmup, 80 epochs, batch 64, val-frac 0.05, seed 42.
#
# Target: the published ~85-88% X-View benchmark. 80.38% did NOT clear it; this is the attempt
# that closes the remaining pyskl gap. Residual known difference: pyskl ensembles 10 test clips,
# we report single-clip (now MEANINGFUL to add, since cyclic finally makes clips differ).
#
# RESUMABLE: re-run this exact command after any interruption. Do not run two copies at once.
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
python src/run_xview_evaluation.py --data data/ntu60_3danno.pkl --models stgcn_full `
  --view-norm --sample cyclic --rot-aug 0.3 `
  --optimizer sgd --lr 0.05 --wd 5e-4 --momentum 0.9 --warmup-epochs 5 `
  --epochs 80 --batch-size 64 --val-frac 0.05 --seed 42 `
  --out outputs/ntu_stgcn_rot

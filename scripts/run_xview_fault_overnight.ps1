# run_xview_fault_overnight.ps1  --  NTU-60 X-View Accuracy x Fault run (Option 1, the thesis figure)
# RESUMABLE: if your PC powers off (load shedding), just RE-RUN this script. Training continues
# from the last COMPLETED epoch (per-epoch checkpoints in outputs/ntu_fault), ST-GCN then EGRU,
# then the node-dropout sweep. Nothing is lost.
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
# 1) train ST-GCN (unmasked) then EGRU (--use-mask = G4 node-dropout), val-selected, checkpointed
python src/run_xview_evaluation.py --data data/ntu60_3danno.pkl --models stgcn egru --use-mask `
  --epochs 60 --batch-size 32 --val-frac 0.05 --seed 42 --out outputs/ntu_fault
# 2) Accuracy x Fault sweep on the saved best weights: k in {0,1,2,4,8} dead nodes
python src/run_xview_evaluation.py --data data/ntu60_3danno.pkl --sweep --sweep-k 0 1 2 4 8 `
  --seed 42 --out outputs/ntu_fault

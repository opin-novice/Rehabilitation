#!/bin/bash
# Resumable NTU X-View fault-curve run. Re-run after any power cut; it resumes.
set -e
export PYTHONUNBUFFERED=1
python src/run_xview_evaluation.py --data data/ntu60_3danno.pkl --models stgcn egru --use-mask \
  --epochs 60 --batch-size 32 --val-frac 0.05 --seed 42 --out outputs/ntu_fault
python src/run_xview_evaluation.py --data data/ntu60_3danno.pkl --sweep --sweep-k 0 1 2 4 8 \
  --seed 42 --out outputs/ntu_fault

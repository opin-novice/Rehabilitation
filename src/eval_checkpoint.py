"""Quick evaluation of a saved checkpoint from train_reproduce.py.

Loads the same data pipeline, reconstructs the model, loads weights,
and evaluates on the test set.

Usage:
    python src/eval_checkpoint.py --ex Kimore_ex1 --ckpt outputs/reproduce/Kimore_ex1/rep1/Kimore_ex1.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from models_curvenet import PointCloudTransformerRegressor
from train_reproduce import (
    extract_body_parts,
    reshape_to_sequences,
    performance_metrics,
    seed_everything,
    load_kimore_exercise,
    NUM_JOINTS,
    NUM_CHANNELS,
    SEQ_LEN,
    NUM_TIMESTEP,
    BODY_PARTS,
)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).float()
            pred = model(x)
            preds.extend(pred.cpu().numpy().flatten())
            trues.extend(y.cpu().numpy().flatten())
    mad, rmse, mse, mape = performance_metrics(trues, preds)
    return mad, rmse, mse, mape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex", default="Kimore_ex1")
    parser.add_argument("--ckpt", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--data_dir", default="KIMORE_processed")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    ex_map = {
        "Kimore_ex1": "Exercise1",
        "Kimore_ex2": "Exercise2",
        "Kimore_ex3": "Exercise3",
        "Kimore_ex4": "Exercise4",
        "Kimore_ex5": "Exercise5",
    }
    exercise_folder = ex_map.get(args.ex, args.ex)
    exercise_dir = os.path.join(args.data_dir, exercise_folder)

    if args.ckpt is None:
        args.ckpt = f"outputs/reproduce/{args.ex}/rep1/{args.ex}.pt"

    print(f"Exercise: {args.ex}")
    print(f"Checkpoint: {args.ckpt}")

    seed_everything(args.seed)

    # Load data
    train_x, test_x, train_y, test_y, sc1, sc2 = load_kimore_exercise(
        exercise_dir, test_size=args.test_size, seed=args.seed
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_x), torch.from_numpy(test_y)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = PointCloudTransformerRegressor(
        seq_len=SEQ_LEN,
        num_joints=NUM_JOINTS,
        num_channels=NUM_CHANNELS,
        dim=256,
        spatial_depth=6,
        temporal_depth=3,
        heads=4,
        dropout=0.1,
        k=20,
        curve_setting="default",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # Load checkpoint
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {args.ckpt}")

    # Evaluate
    loss_fn = nn.HuberLoss(reduction="mean", delta=0.1)
    mad, rmse, mse, mape = evaluate(model, test_loader, device, loss_fn)

    print(f"\n{'='*50}")
    print(f"  {args.ex} -- Checkpoint Evaluation")
    print(f"{'='*50}")
    print(f"  MAD:  {mad:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"{'='*50}")

    # Also evaluate on train set for reference
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    train_mad, train_rmse, train_mse, train_mape = evaluate(model, train_loader, device, loss_fn)
    print(f"\n  Train MAD: {train_mad:.4f}  RMSE: {train_rmse:.4f}  MAPE: {train_mape:.2f}%")

    # Paper reference
    paper_ref = {
        "Kimore_ex1": {"mad": 0.071, "rmse": 0.105, "mape": 20.80},
        "Kimore_ex2": {"mad": 0.075, "rmse": 0.106, "mape": 22.54},
        "Kimore_ex3": {"mad": 0.067, "rmse": 0.101, "mape": 21.78},
        "Kimore_ex4": {"mad": 0.070, "rmse": 0.103, "mape": 22.44},
        "Kimore_ex5": {"mad": 0.059, "rmse": 0.090, "mape": 19.03},
    }
    if args.ex in paper_ref:
        ref = paper_ref[args.ex]
        print(f"\n  Paper ref: MAD={ref['mad']:.3f}, RMSE={ref['rmse']:.3f}, MAPE={ref['mape']:.2f}%")
        print(f"  Gap: MAD={abs(mad-ref['mad']):.4f}, RMSE={abs(rmse-ref['rmse']):.4f}, MAPE={abs(mape-ref['mape']):.2f}%")


if __name__ == "__main__":
    main()

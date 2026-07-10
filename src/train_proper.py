"""Proper ML training script with crash-resilient checkpointing.

Key differences from train_reproduce.py:
  - Subject-grouped split (prevents data leakage)
  - StandardScaler fit on TRAIN only
  - Validation-set model selection (no test set leakage)
  - Same architecture and hyperparameters

Crash-resilient: saves full state every --save_every epochs.
Resume with --resume to continue from last checkpoint.

Usage:
    python src/train_proper.py --ex Kimore_ex1 --reps 10
    python src/train_proper.py --ex Kimore_ex1 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models_curvenet import PointCloudTransformerRegressor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BODY_PARTS = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36,
    40, 44, 48, 52, 56, 60, 64, 68, 72, 76,
    80, 84, 88, 92, 96,
]
NUM_JOINTS = 25
NUM_CHANNELS = 3
SEQ_LEN = 100
NUM_TIMESTEP = 100


def extract_body_parts(raw):
    n_rows = raw.shape[0]
    n_cols = len(BODY_PARTS) * NUM_CHANNELS
    extracted = np.zeros((n_rows, n_cols), dtype=np.float32)
    for row in range(n_rows):
        counter = 0
        for part in BODY_PARTS:
            for ch in range(NUM_CHANNELS):
                extracted[row, counter + ch] = raw[row, part + ch]
            counter += NUM_CHANNELS
    return extracted


def reshape_to_sequences(scaled_x, n_samples, seq_len, num_joints, num_channels):
    x_4d = np.zeros((n_samples, seq_len, num_joints, num_channels), dtype=np.float32)
    for b in range(n_samples):
        for t in range(seq_len):
            for j in range(num_joints):
                for c in range(num_channels):
                    x_4d[b, t, j, c] = scaled_x[t + b * seq_len, c + j * num_channels]
    return x_4d


def mean_absolute_percentage_error(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def performance_metrics(y_true, y_pred):
    from math import sqrt
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mse, mape


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Data loading (proper ML methodology)
# ---------------------------------------------------------------------------

def load_kimore_exercise_proper(exercise_dir, test_size=0.2, seed=420, subject_ids_csv=None):
    raw_x = pd.read_csv(os.path.join(exercise_dir, "Train_X.csv"), header=None).values
    raw_y = pd.read_csv(os.path.join(exercise_dir, "Train_Y.csv"), header=None).values

    n_total = raw_x.shape[0]
    n_samples = n_total // NUM_TIMESTEP
    print(f"  Loaded {n_total} rows, {n_samples} samples")

    xyz = extract_body_parts(raw_x)

    if subject_ids_csv and os.path.exists(subject_ids_csv):
        subject_ids = pd.read_csv(subject_ids_csv, header=None).values.squeeze()
    else:
        sid_path = os.path.join(exercise_dir, "subject_ids.csv")
        if os.path.exists(sid_path):
            subject_ids = pd.read_csv(sid_path, header=None).values.squeeze()
        else:
            subject_ids = np.arange(n_samples) // 5

    print(f"  Unique subjects: {len(np.unique(subject_ids))}")

    # Subject-grouped split: 80% train, 10% val, 10% test
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    trainval_idx, test_idx = next(splitter1.split(np.arange(n_samples), groups=subject_ids))

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.111, random_state=seed)
    train_idx, val_idx = next(splitter2.split(trainval_idx, groups=subject_ids[trainval_idx]))

    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Train subjects: {len(np.unique(subject_ids[train_idx]))}")
    print(f"  Val subjects: {len(np.unique(subject_ids[val_idx]))}")
    print(f"  Test subjects: {len(np.unique(subject_ids[test_idx]))}")

    def flatten_to_rows(idx):
        rows = np.concatenate([np.arange(i * NUM_TIMESTEP, (i + 1) * NUM_TIMESTEP) for i in idx])
        return xyz[rows], raw_y[idx]

    train_rows_x, train_y = flatten_to_rows(train_idx)
    val_rows_x, val_y = flatten_to_rows(val_idx)
    test_rows_x, test_y = flatten_to_rows(test_idx)

    sc1 = StandardScaler()
    sc2 = StandardScaler()

    train_x_scaled = sc1.fit_transform(train_rows_x).astype(np.float32)
    val_x_scaled = sc1.transform(val_rows_x).astype(np.float32)
    test_x_scaled = sc1.transform(test_rows_x).astype(np.float32)

    train_y_scaled = sc2.fit_transform(train_y.reshape(-1, 1)).reshape(-1).astype(np.float32)
    val_y_scaled = sc2.transform(val_y.reshape(-1, 1)).reshape(-1).astype(np.float32)
    test_y_scaled = sc2.transform(test_y.reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_x_4d = reshape_to_sequences(train_x_scaled, len(train_idx), NUM_TIMESTEP, NUM_JOINTS, NUM_CHANNELS)
    val_x_4d = reshape_to_sequences(val_x_scaled, len(val_idx), NUM_TIMESTEP, NUM_JOINTS, NUM_CHANNELS)
    test_x_4d = reshape_to_sequences(test_x_scaled, len(test_idx), NUM_TIMESTEP, NUM_JOINTS, NUM_CHANNELS)

    print(f"  Train Y: [{train_y_scaled.min():.4f}, {train_y_scaled.max():.4f}]")
    print(f"  Val Y:   [{val_y_scaled.min():.4f}, {val_y_scaled.max():.4f}]")
    print(f"  Test Y:  [{test_y_scaled.min():.4f}, {test_y_scaled.max():.4f}]")

    return (
        train_x_4d, val_x_4d, test_x_4d,
        train_y_scaled.reshape(-1, 1), val_y_scaled.reshape(-1, 1), test_y_scaled.reshape(-1, 1),
        sc1, sc2,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers (same as train_reproduce.py)
# ---------------------------------------------------------------------------

def _ckpt_paths(save_path, ex):
    return (
        os.path.join(save_path, f"{ex}_resume.pt"),
        os.path.join(save_path, f"{ex}.pt"),
    )


def save_checkpoint(state, save_path, ex):
    os.makedirs(save_path, exist_ok=True)
    resume_ckpt, best_ckpt = _ckpt_paths(save_path, ex)
    tmp = resume_ckpt + ".tmp"
    torch.save(state, tmp)
    shutil.move(tmp, resume_ckpt)
    if "best_state_dict" in state:
        tmp2 = best_ckpt + ".tmp"
        torch.save(state["best_state_dict"], tmp2)
        shutil.move(tmp2, best_ckpt)


def save_history(history, save_path, ex):
    path = os.path.join(save_path, f"{ex}_history.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    shutil.move(tmp, path)


def load_checkpoint(save_path, ex):
    resume_ckpt, _ = _ckpt_paths(save_path, ex)
    if os.path.exists(resume_ckpt):
        print(f"  Found checkpoint: {resume_ckpt}")
        return torch.load(resume_ckpt, map_location="cpu", weights_only=False)
    return None


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_mad(model, loader, device, loss_fn, scaler_y=None):
    """Evaluate model on SCALED values (used during training for model selection)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).float()
            pred = model(x)
            preds.extend(pred.cpu().numpy().flatten())
            trues.extend(y.cpu().numpy().flatten())
    return performance_metrics(trues, preds)


def evaluate_original_scale(model, loader, device, scaler_y):
    """Evaluate model on ORIGINAL Y values (inverse-transformed).
    
    Matches the original repo's eval.py which does:
        preds = data_loader.sc2.inverse_transform(preds)
        trues = data_loader.sc2.inverse_transform(trues)
    
    The paper's reported metrics are in original Y units.
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).float()
            pred = model(x)
            preds.extend(pred.cpu().numpy().flatten())
            trues.extend(y.cpu().numpy().flatten())
    
    preds = np.array(preds).reshape(-1, 1)
    trues = np.array(trues).reshape(-1, 1)
    
    if scaler_y is not None:
        preds = scaler_y.inverse_transform(preds)
        trues = scaler_y.inverse_transform(trues)
    
    return performance_metrics(trues, preds)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_proper(
    epochs, train_loader, val_loader, test_loader,
    save_path, ex="model", lr=1e-4, weight_decay=0.0,
    huber_delta=0.1, device=None, save_every=100, resume=False,
    scaler_y=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_path, exist_ok=True)

    model = PointCloudTransformerRegressor(
        seq_len=SEQ_LEN, num_joints=NUM_JOINTS, num_channels=NUM_CHANNELS,
        dim=256, spatial_depth=6, temporal_depth=3, heads=4,
        dropout=0.1, k=20, curve_setting="default",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(reduction="mean", delta=huber_delta)

    # Resume state
    start_epoch = 1
    best_val_mad = float("inf")
    best_epoch = 0
    best_state_dict = None
    history = {"epoch": [], "train_loss": [],
               "val_mad": [], "val_rmse": [], "val_mape": [],
               "test_mad": [], "test_rmse": [], "test_mape": []}
    total_time = 0.0

    if resume:
        ckpt = load_checkpoint(save_path, ex)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_mad = ckpt["best_val_mad"]
            best_epoch = ckpt["best_epoch"]
            best_state_dict = ckpt.get("best_state_dict")
            total_time = ckpt.get("total_time", 0.0)
            hist_path = os.path.join(save_path, f"{ex}_history.json")
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    history = json.load(f)
            print(f"  Resumed from epoch {ckpt['epoch']} | best val MAD={best_val_mad:.4f} at epoch {best_epoch}")
        else:
            print("  No checkpoint found, starting fresh.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,} | Device: {device}")
    print(f"  Training epochs {start_epoch}-{epochs} (save_every={save_every})")

    wall_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        train_preds, train_trues = [], []

        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).float()
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            train_preds.extend(pred.detach().cpu().numpy().flatten())
            train_trues.extend(y.detach().cpu().numpy().flatten())

        train_loss = total_loss / max(n, 1)
        train_mad = performance_metrics(train_trues, train_preds)[0]

        val_mad, val_rmse, val_mse, val_mape = evaluate_mad(val_loader, device, loss_fn)
        test_mad, test_rmse, test_mse, test_mape = evaluate_mad(test_loader, device, loss_fn)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_mad"].append(val_mad)
        history["val_rmse"].append(val_rmse)
        history["val_mape"].append(val_mape)
        history["test_mad"].append(test_mad)
        history["test_rmse"].append(test_rmse)
        history["test_mape"].append(test_mape)

        # Best on VALIDATION set
        if val_mad < best_val_mad:
            best_val_mad = val_mad
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  >> New best val MAD={best_val_mad:.4f} at epoch {best_epoch}")

        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - wall_start + total_time
            print(
                f"  [epoch {epoch}/{epochs}] train_loss={train_loss:.4f} "
                f"val MAD={val_mad:.4f} test MAD={test_mad:.4f} RMSE={test_rmse:.4f} "
                f"MAPE={test_mape:.2f}% | {elapsed:.0f}s"
            )

        if epoch % save_every == 0 or epoch == epochs:
            state = {
                "epoch": epoch,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mad": best_val_mad,
                "best_epoch": best_epoch,
                "best_state_dict": best_state_dict,
                "total_time": time.time() - wall_start + total_time,
            }
            save_checkpoint(state, save_path, ex)
            save_history(history, save_path, ex)

    elapsed = time.time() - wall_start + total_time
    print(f"\n  Done in {elapsed:.0f}s. Best val MAD={best_val_mad:.4f} at epoch {best_epoch}")

    save_history(history, save_path, ex)

    # Load best weights
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    # Final test evaluation (original-scale metrics, matching paper)
    test_mad_orig, test_rmse_orig, test_mse_orig, test_mape_orig = evaluate_original_scale(
        model, test_loader, device, scaler_y=scaler_y
    )
    test_mad_scaled, test_rmse_scaled, test_mse_scaled, test_mape_scaled = evaluate_mad(
        model, test_loader, device, loss_fn
    )
    print(f"  Best model test (original): MAD={test_mad_orig:.4f}, RMSE={test_rmse_orig:.4f}, MAPE={test_mape_orig:.2f}%")
    print(f"  Best model test (scaled):   MAD={test_mad_scaled:.4f}, RMSE={test_rmse_scaled:.4f}, MAPE={test_mape_scaled:.2f}%")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Proper ML training for KIMORE.")
    parser.add_argument("--ex", default="Kimore_ex1")
    parser.add_argument("--data_dir", default="KIMORE_processed")
    parser.add_argument("--out_dir", default="outputs/proper")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--huber_delta", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start_rep", type=int, default=0)
    args = parser.parse_args()

    ex_map = {
        "Kimore_ex1": "Exercise1", "Kimore_ex2": "Exercise2",
        "Kimore_ex3": "Exercise3", "Kimore_ex4": "Exercise4",
        "Kimore_ex5": "Exercise5",
    }
    exercise_folder = ex_map.get(args.ex, args.ex)
    exercise_dir = os.path.join(args.data_dir, exercise_folder)

    if not os.path.exists(exercise_dir):
        print(f"Error: {exercise_dir} not found")
        sys.exit(1)

    print(f"=== Proper ML: {args.ex} | {args.reps} reps | {args.epochs} epochs ===")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Save every: {args.save_every} epochs")
    print(f"  Resume: {args.resume}")
    print()

    all_results = []
    results_path = os.path.join(args.out_dir, args.ex, "all_results.json")

    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} existing results")

    for rep in range(args.start_rep, args.reps):
        rep_seed = args.seed + rep
        out_dir = os.path.join(args.out_dir, args.ex, f"rep{rep + 1}")

        rep_result_file = os.path.join(out_dir, "result.json")
        if args.resume and os.path.exists(rep_result_file):
            with open(rep_result_file) as f:
                result = json.load(f)
            all_results.append(result)
            print(f"\n--- Rep {rep+1}/{args.reps}: already done (MAD={result['mad']:.4f}) ---")
            continue

        print(f"\n--- Repetition {rep+1}/{args.reps} (seed={rep_seed}) ---")
        seed_everything(rep_seed)

        train_x, val_x, test_x, train_y, val_y, test_y, _, sc_y = load_kimore_exercise_proper(
            exercise_dir, test_size=args.test_size, seed=rep_seed,
        )

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=args.batch_size, shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
            batch_size=args.batch_size, shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
            batch_size=args.batch_size, shuffle=False,
        )

        model = train_proper(
            epochs=args.epochs, train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, save_path=out_dir, ex=args.ex, lr=args.lr,
            weight_decay=args.weight_decay, huber_delta=args.huber_delta,
            save_every=args.save_every, resume=args.resume, scaler_y=sc_y,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Final evaluation: original-scale metrics (matches paper)
        mad_orig, rmse_orig, mse_orig, mape_orig = evaluate_original_scale(
            model, test_loader, device, scaler_y=sc_y
        )
        loss_fn = nn.HuberLoss(reduction="mean", delta=args.huber_delta)
        mad_scaled, rmse_scaled, mse_scaled, mape_scaled = evaluate_mad(
            model, test_loader, device, loss_fn
        )

        result = {
            "rep": rep + 1, "seed": rep_seed,
            # Original-scale metrics (matches paper)
            "mad": mad_orig, "rmse": rmse_orig, "mse": mse_orig, "mape": mape_orig,
            # Scaled metrics (for debugging)
            "mad_scaled": mad_scaled, "rmse_scaled": rmse_scaled,
            "mse_scaled": mse_scaled, "mape_scaled": mape_scaled,
        }
        all_results.append(result)

        with open(rep_result_file, "w") as f:
            json.dump(result, f, indent=2)

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  Final test (original): MAD={mad_orig:.4f}, RMSE={rmse_orig:.4f}, MAPE={mape_orig:.2f}%")
        print(f"  Final test (scaled):   MAD={mad_scaled:.4f}, RMSE={rmse_scaled:.4f}, MAPE={mape_scaled:.2f}%")

    # Aggregate (original-scale metrics, matching paper)
    print(f"\n{'='*60}")
    mads = [r["mad"] for r in all_results]
    rmses = [r["rmse"] for r in all_results]
    mapes = [r["mape"] for r in all_results]
    mads_s = [r["mad_scaled"] for r in all_results]

    print(f"=== {args.ex} ({len(all_results)} reps) ===")
    print(f"  MAD (original):  {np.mean(mads):.4f} +/- {np.std(mads):.4f}")
    print(f"  RMSE (original): {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"  MAPE (original): {np.mean(mapes):.2f}% +/- {np.std(mapes):.2f}%")
    print(f"  MAD (scaled):    {np.mean(mads_s):.4f} +/- {np.std(mads_s):.4f}")

    summary = {
        "exercise": args.ex, "reps": len(all_results),
        # Original-scale metrics (matches paper)
        "mad_mean": float(np.mean(mads)), "mad_std": float(np.std(mads)),
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "mape_mean": float(np.mean(mapes)), "mape_std": float(np.std(mapes)),
        # Scaled metrics (for debugging)
        "mad_scaled_mean": float(np.mean(mads_s)), "mad_scaled_std": float(np.std(mads_s)),
        "all_results": all_results,
    }
    summary_path = os.path.join(args.out_dir, args.ex, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()

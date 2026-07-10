"""Faithful reproduction of Rafat et al. (JBHI 2026) KIMORE training pipeline.

Matches the original repo's exact setup:
  - Global StandardScaler (fit on ALL data, not just train)
  - Simple train_test_split (not subject-grouped)
  - AdamW lr=0.0001, weight_decay=0.0
  - HuberLoss(delta=0.1)
  - Batch size=1, 2000 epochs
  - No scheduler, no gradient clipping
  - Test-set model selection (best epoch on test set)
  - MAD, MAPE, RMSE metrics

Crash-resilient: saves full state every --save_every epochs.
Resume with --resume to continue from last checkpoint.

Usage:
    python src/train_reproduce.py --ex Kimore_ex1 --reps 10
    python src/train_reproduce.py --ex Kimore_ex1 --resume
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
from sklearn.model_selection import train_test_split
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


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def extract_body_parts(raw: np.ndarray) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kimore_exercise(exercise_dir, test_size=0.2, seed=420):
    train_x = pd.read_csv(os.path.join(exercise_dir, "Train_X.csv"), header=None).values
    train_y = pd.read_csv(os.path.join(exercise_dir, "Train_Y.csv"), header=None).values

    n_total = train_x.shape[0]
    n_samples = n_total // NUM_TIMESTEP
    print(f"  Loaded {n_total} rows, {n_samples} samples")

    X_train = extract_body_parts(train_x)
    y_flat = train_y.reshape(-1, 1)

    # Scale X on ALL data (matches original repo)
    sc1 = StandardScaler()
    X_scaled = sc1.fit_transform(X_train)

    # Reshape to sequences
    X_4d = reshape_to_sequences(X_scaled, n_samples, NUM_TIMESTEP, NUM_JOINTS, NUM_CHANNELS)
    print(f"  Reshaped to: {X_4d.shape}")

    # Split FIRST, then scale Y on train only (matches original repo)
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_4d, y_flat, test_size=test_size, random_state=seed
    )

    # Fit Y scaler on TRAIN only (no data leakage)
    sc2 = StandardScaler()
    y_train_scaled = sc2.fit_transform(y_train_split).astype(np.float32)
    y_test_scaled = sc2.transform(y_test_split).astype(np.float32)

    print(f"  Train: {len(X_train_split)}, Test: {len(X_test_split)}")
    print(f"  Train Y: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
    print(f"  Test Y:  [{y_test_scaled.min():.4f}, {y_test_scaled.max():.4f}]")

    return X_train_split, X_test_split, y_train_scaled, y_test_scaled, sc1, sc2


# ---------------------------------------------------------------------------
# Checkpoint helpers  (crash-resilient)
# ---------------------------------------------------------------------------

def _ckpt_paths(save_path, ex):
    """Return (resume_ckpt, best_ckpt) paths."""
    return (
        os.path.join(save_path, f"{ex}_resume.pt"),
        os.path.join(save_path, f"{ex}.pt"),
    )


def save_checkpoint(state, save_path, ex):
    """Save full training state atomically (write tmp, then rename)."""
    os.makedirs(save_path, exist_ok=True)
    resume_ckpt, best_ckpt = _ckpt_paths(save_path, ex)

    # Save resume checkpoint (full state)
    tmp = resume_ckpt + ".tmp"
    torch.save(state, tmp)
    shutil.move(tmp, resume_ckpt)

    # Save best model weights only (for easy loading)
    if "best_state_dict" in state:
        tmp2 = best_ckpt + ".tmp"
        torch.save(state["best_state_dict"], tmp2)
        shutil.move(tmp2, best_ckpt)


def save_history(history, save_path, ex):
    """Save history JSON atomically."""
    path = os.path.join(save_path, f"{ex}_history.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    shutil.move(tmp, path)


def load_checkpoint(save_path, ex):
    """Load resume checkpoint. Returns state dict or None."""
    resume_ckpt, _ = _ckpt_paths(save_path, ex)
    if os.path.exists(resume_ckpt):
        print(f"  Found checkpoint: {resume_ckpt}")
        return torch.load(resume_ckpt, map_location="cpu", weights_only=False)
    return None


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_mad(model, loader, device, loss_fn, scaler_y=None):
    """Evaluate model on SCALED values (used during training for best-model selection)."""
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
    
    This matches the original repo's eval.py which does:
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

def train(
    epochs,
    train_loader,
    test_loader,
    save_path,
    ex="model",
    lr=1e-4,
    weight_decay=0.0,
    huber_delta=0.1,
    device=None,
    save_every=100,
    resume=False,
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
    best_mad = float("inf")
    best_epoch = 0
    best_state_dict = None
    history = {"epoch": [], "train_loss": [], "test_mad": [],
               "test_rmse": [], "test_mse": [], "test_mape": []}
    total_time = 0.0

    if resume:
        ckpt = load_checkpoint(save_path, ex)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_mad = ckpt["best_mad"]
            best_epoch = ckpt["best_epoch"]
            best_state_dict = ckpt.get("best_state_dict")
            total_time = ckpt.get("total_time", 0.0)
            # Restore history
            hist_path = os.path.join(save_path, f"{ex}_history.json")
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    history = json.load(f)
            print(f"  Resumed from epoch {ckpt['epoch']} | best MAD={best_mad:.4f} at epoch {best_epoch}")
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

        test_mad, test_rmse, test_mse, test_mape = evaluate_mad(
            model, test_loader, device, loss_fn, scaler_y=scaler_y
        )

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_mad"].append(test_mad)
        history["test_rmse"].append(test_rmse)
        history["test_mse"].append(test_mse)
        history["test_mape"].append(test_mape)

        # Track best
        if test_mad < best_mad:
            best_mad = test_mad
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  >> New best MAD={best_mad:.4f} at epoch {best_epoch}")

        # Periodic logging
        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - wall_start + total_time
            print(
                f"  [epoch {epoch}/{epochs}] train_loss={train_loss:.4f} "
                f"test MAD={test_mad:.4f} RMSE={test_rmse:.4f} MAPE={test_mape:.2f}% "
                f"| elapsed {elapsed:.0f}s"
            )

        # Save state every save_every epochs (crash resilience)
        if epoch % save_every == 0 or epoch == epochs:
            state = {
                "epoch": epoch,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mad": best_mad,
                "best_epoch": best_epoch,
                "best_state_dict": best_state_dict,
                "total_time": time.time() - wall_start + total_time,
                "args": {
                    "ex": ex, "lr": lr, "weight_decay": weight_decay,
                    "huber_delta": huber_delta, "epochs": epochs,
                },
            }
            save_checkpoint(state, save_path, ex)
            save_history(history, save_path, ex)

    elapsed = time.time() - wall_start + total_time
    print(f"\n  Done in {elapsed:.0f}s. Best test MAD={best_mad:.4f} at epoch {best_epoch}")

    # Final save
    save_history(history, save_path, ex)

    # Load best weights back into model for return
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reproduce Rafat et al. KIMORE results.")
    parser.add_argument("--ex", default="Kimore_ex1")
    parser.add_argument("--data_dir", default="KIMORE_processed")
    parser.add_argument("--out_dir", default="outputs/reproduce")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--huber_delta", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--start_rep", type=int, default=0,
                        help="Rep to start from (0-indexed, for resume)")
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

    print(f"=== {args.ex} | {args.reps} reps | {args.epochs} epochs ===")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Save every: {args.save_every} epochs")
    print(f"  Resume: {args.resume}")
    print()

    all_results = []
    results_path = os.path.join(args.out_dir, args.ex, "all_results.json")

    # Load existing results if resuming
    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} existing results from {results_path}")

    for rep in range(args.start_rep, args.reps):
        rep_seed = args.seed + rep
        out_dir = os.path.join(args.out_dir, args.ex, f"rep{rep + 1}")

        # Check if this rep already has results
        rep_result_file = os.path.join(out_dir, "result.json")
        if args.resume and os.path.exists(rep_result_file):
            with open(rep_result_file) as f:
                result = json.load(f)
            all_results.append(result)
            print(f"\n--- Rep {rep+1}/{args.reps}: already done (MAD={result['mad']:.4f}) ---")
            continue

        print(f"\n--- Repetition {rep+1}/{args.reps} (seed={rep_seed}) ---")
        seed_everything(rep_seed)

        train_x, test_x, train_y, test_y, sc_x, sc_y = load_kimore_exercise(
            exercise_dir, test_size=args.test_size, seed=rep_seed
        )

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=args.batch_size, shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
            batch_size=args.batch_size, shuffle=False,
        )

        model = train(
            epochs=args.epochs, train_loader=train_loader, test_loader=test_loader,
            save_path=out_dir, ex=args.ex, lr=args.lr, weight_decay=args.weight_decay,
            huber_delta=args.huber_delta, save_every=args.save_every,
            resume=args.resume, scaler_y=sc_y,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Final evaluation: original-scale metrics (matches paper's reported values)
        mad_orig, rmse_orig, mse_orig, mape_orig = evaluate_original_scale(
            model, test_loader, device, scaler_y=sc_y
        )
        
        # Also compute scaled metrics for reference
        loss_fn = nn.HuberLoss(reduction="mean", delta=args.huber_delta)
        mad_scaled, rmse_scaled, mse_scaled, mape_scaled = evaluate_mad(
            model, test_loader, device, loss_fn, scaler_y=sc_y
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

        # Save per-rep result
        with open(rep_result_file, "w") as f:
            json.dump(result, f, indent=2)

        # Save cumulative results
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  Final: MAD={mad_orig:.4f} RMSE={rmse_orig:.4f} MAPE={mape_orig:.2f}% (original scale)")
        print(f"         MAD={mad_scaled:.4f} RMSE={rmse_scaled:.4f} MAPE={mape_scaled:.2f}% (scaled)")

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

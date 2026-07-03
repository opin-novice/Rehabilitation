"""Standalone evaluation script.

Loads the model checkpoint (which includes the serialized ScalerBundle),
runs inference on the validation split, and prints metrics + saves plots.

IMPORTANT: This script loads the scaler from the checkpoint rather than
re-fitting it — avoiding the metric corruption bug present in the original
guide's evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from rehab_dataset import ScalerBundle, make_dataloaders
from models import build_model
from visualize import plot_prediction_scatter, plot_residuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved rehabilitation model.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="outputs/eval")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args: dict = ckpt["args"]

    # --- Load ScalerBundle from sidecar file (KEY FIX) ---
    scalers_path = str(Path(args.checkpoint).parent / "scalers.pkl")
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"scalers.pkl not found alongside checkpoint: {scalers_path}")
    bundle: ScalerBundle = joblib.load(scalers_path)

    # --- Rebuild dataloaders using saved hyperparameters ---
    subject_ids_csv_path = os.path.join(args.data_dir, "subject_ids.csv")
    subject_ids_csv = subject_ids_csv_path if os.path.exists(subject_ids_csv_path) else None

    _, val_loader, _ = make_dataloaders(
        data_dir=args.data_dir,
        seq_len=saved_args.get("seq_len", 100),
        batch_size=args.batch_size,
        test_size=saved_args.get("test_size", 0.2),
        reps_per_subject=saved_args.get("reps_per_subject", 5),
        subject_ids_csv=subject_ids_csv,
        seed=saved_args.get("seed", 145),
        num_workers=args.num_workers,
    )

    # --- Rebuild model ---
    model = build_model(
        model_type=saved_args.get("model", "transformer"),
        seq_len=saved_args.get("seq_len", 100),
        dropout=saved_args.get("dropout", 0.1),
        joint_dim=saved_args.get("joint_dim", 128),
        d_model=saved_args.get("d_model", 256),
        spatial_heads=saved_args.get("spatial_heads", 4),
        spatial_layers=saved_args.get("spatial_layers", 2),
        temporal_heads=saved_args.get("temporal_heads", 4),
        temporal_layers=saved_args.get("temporal_layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- Inference ---
    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device).float()
            out = model(x)
            preds_list.append(out.cpu().numpy())
            targets_list.append(y.numpy())

    preds_scaled = np.concatenate(preds_list).reshape(-1)
    targets_scaled = np.concatenate(targets_list).reshape(-1)

    # --- Inverse transform using checkpoint scalers ---
    preds = bundle.y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
    targets = bundle.y_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds)) if len(targets) > 1 else float("nan")

    metrics = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch", "?"),
        "val_rmse": round(rmse, 6),
        "val_mae": round(mae, 6),
        "val_r2": round(r2, 6),
        "n_val_samples": len(targets),
    }
    print(json.dumps(metrics, indent=2))

    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_prediction_scatter(
        targets, preds,
        os.path.join(args.out_dir, "prediction_scatter.png"),
        title=f"Eval — RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}",
    )
    plot_residuals(
        targets, preds,
        os.path.join(args.out_dir, "residuals.png"),
    )
    print(f"Plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()

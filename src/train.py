"""Training script for rehabilitation exercise quality assessment.

Key engineering choices vs. original repo:
  - HuberLoss(delta=0.1): matches original repo; more robust than MSE for outlier scores.
  - Subject-grouped train/val split: prevents subject-level data leakage.
  - ScalerBundle serialized in checkpoint: evaluate.py can load it without re-fitting.
  - Structured JSON logging to file + console.
  - Shape assertions via model forward passes (see models.py).
  - Optional W&B logging via --wandb flag (requires: pip install wandb).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Local imports — run from src/ or adjust PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))
from rehab_dataset import ScalerBundle, make_dataloaders
from models import build_model, count_parameters
from visualize import plot_prediction_scatter, plot_training_curves


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

class _JsonLineFormatter(logging.Formatter):
    """Emit one JSON object per log record for easy downstream parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "metrics"):
            payload.update(record.metrics)
        return json.dumps(payload)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("rehab")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console: human-readable
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

    # File: JSON lines
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JsonLineFormatter())
    logger.addHandler(fh)
    return logger


def log_metrics(logger: logging.Logger, metrics: dict) -> None:
    record = logging.LogRecord(
        name="rehab", level=logging.INFO, pathname="", lineno=0,
        msg=json.dumps(metrics), args=(), exc_info=None,
    )
    record.metrics = metrics
    logger.handle(record)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def inverse_y(bundle: ScalerBundle, arr: np.ndarray) -> np.ndarray:
    return bundle.y_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(-1)


def compute_metrics(
    bundle: ScalerBundle,
    preds_scaled: np.ndarray,
    targets_scaled: np.ndarray,
) -> dict[str, float]:
    preds = inverse_y(bundle, preds_scaled)
    targets = inverse_y(bundle, targets_scaled)
    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds)) if len(targets) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    bundle: ScalerBundle,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds_list.append(out.cpu().numpy())
            targets_list.append(y.cpu().numpy())

    preds = np.concatenate(preds_list).reshape(-1)
    targets = np.concatenate(targets_list).reshape(-1)
    avg_loss = total_loss / max(n, 1)
    metrics = compute_metrics(bundle, preds, targets)
    return avg_loss, metrics, preds, targets


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train rehabilitation quality regressor.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="outputs/exp1")
    parser.add_argument("--model", choices=["e0_mlp", "transformer"], default="transformer")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--huber_delta", type=float, default=0.1,
                        help="Huber loss delta (matches original repo).")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--joint_dim", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--spatial_heads", type=int, default=4)
    parser.add_argument("--spatial_layers", type=int, default=2)
    parser.add_argument("--temporal_heads", type=int, default=4)
    parser.add_argument("--temporal_layers", type=int, default=3)
    parser.add_argument("--reps_per_subject", type=int, default=5,
                        help="Repetitions per subject for subject-aware split.")
    parser.add_argument("--subject_ids_csv", default="",
                        help="Optional path to subject_ids.csv; inferred if absent.")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=145)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Mixed-precision training (CUDA only).")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--clip_grad", type=float, default=1.0)
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(out_dir / "train.log"))
    logger.info(f"Args: {vars(args)}")

    # --- W&B (optional) ---
    wb_run: Optional[object] = None
    if args.wandb:
        try:
            import wandb
            wb_run = wandb.init(project="rehab_transformer", config=vars(args))
            logger.info("W&B run initialised.")
        except ImportError:
            logger.warning("wandb not installed; skipping. pip install wandb to enable.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Data ---
    subject_ids_csv: Optional[str] = args.subject_ids_csv or None
    train_loader, val_loader, bundle = make_dataloaders(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        test_size=args.test_size,
        reps_per_subject=args.reps_per_subject,
        subject_ids_csv=subject_ids_csv,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # --- Model ---
    model = build_model(
        model_type=args.model,
        seq_len=args.seq_len,
        dropout=args.dropout,
        joint_dim=args.joint_dim,
        d_model=args.d_model,
        spatial_heads=args.spatial_heads,
        spatial_layers=args.spatial_layers,
        temporal_heads=args.temporal_heads,
        temporal_layers=args.temporal_layers,
    ).to(device)

    n_params = count_parameters(model)
    logger.info(f"Model: {args.model} | Trainable params: {n_params:,}")

    # --- Optimiser / scheduler ---
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    loss_fn = nn.HuberLoss(reduction="mean", delta=args.huber_delta)

    use_amp = args.amp and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Save args + config alongside checkpoint ---
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_rmse": [], "val_mae": [], "val_r2": []
    }
    best_rmse = float("inf")
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
                loss = loss_fn(out, y)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler_amp.step(opt)
            scaler_amp.update()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        scheduler.step()
        train_loss = total_loss / max(n, 1)
        val_loss, val_metrics, preds, targets = evaluate(model, val_loader, bundle, device, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            **{k: round(v, 6) for k, v in val_metrics.items()},
            "lr": scheduler.get_last_lr()[0],
        }
        log_metrics(logger, epoch_metrics)

        if wb_run is not None:
            import wandb
            wandb.log(epoch_metrics)

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            # Save model tensors (weights-only .pt) and scalers separately
            # (sklearn objects can't be embedded in weights-only torch.load).
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_rmse": best_rmse,
                },
                str(out_dir / "best_model.pt"),
            )
            joblib.dump(bundle, str(out_dir / "scalers.pkl"))
            plot_prediction_scatter(
                inverse_y(bundle, targets),
                inverse_y(bundle, preds),
                str(out_dir / "prediction_scatter.png"),
                title=f"Epoch {epoch} (best RMSE={best_rmse:.4f})",
            )
            logger.info(f"  >> New best RMSE={best_rmse:.4f} -- checkpoint saved.")

        plot_training_curves(history, str(out_dir / "training_curves.png"))

    elapsed = time.perf_counter() - t0
    np.save(str(out_dir / "history.npy"), history)
    logger.info(
        f"Done in {elapsed:.1f}s | Best val RMSE={best_rmse:.4f} | "
        f"Outputs in {args.out_dir}"
    )

    if wb_run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

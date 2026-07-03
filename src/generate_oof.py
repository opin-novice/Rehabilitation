"""Generate out-of-fold prediction CSVs from existing trained LOSO checkpoints.

Avoids full retraining by loading each fold's best_model.pt and scalers.pkl,
reconstructing the identical val split (deterministic: StratifiedGroupKFold with
shuffle=False), running inference on the val set, and writing:

  <exp_dir>/fold_X/oof_predictions.csv   — per-fold CSV
  <exp_dir>/oof_predictions_all.csv      — all folds concatenated

Columns: subject_id | exercise_id | y_true | y_pred | abs_error | fold

Usage:
  python src/generate_oof.py --exp_dir outputs/loso_lstm
  python src/generate_oof.py --exp_dir outputs/loso_pooled --split_method group
  python src/generate_oof.py --all      # runs all registered experiments
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from rehab_dataset import NUM_CHANNELS, NUM_JOINTS, ScalerBundle
from models import build_model
from models_stgcn import STGCNRegressor, LSTMRegressor, TCNRegressor, SCTRegressor

SEQ_LEN = 100
NUM_EXERCISES = 15   # KIMORE 0-4, UI-PRMD 5-14

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENTS: list[tuple[str, str, str]] = [
    ("Exp E (Transformer)",      "outputs/loso_multitask_uiprmd_d128",      "stratified"),
    ("LSTM baseline",            "outputs/loso_lstm",                        "stratified"),
    ("ST-GCN",                   "outputs/loso_stgcn",                       "stratified"),
    ("GraphTransformer",         "outputs/loso_graph_transformer",           "stratified"),
    ("GraphTransformer (no bias)","outputs/loso_graph_transformer_no_bias",  "stratified"),
    ("TCN",                      "outputs/loso_tcn",                         "stratified"),
    ("SCT",                      "outputs/loso_sct",                         "stratified"),
    ("Baseline GroupKFold",      "outputs/loso_pooled",                      "group"),
    ("Exp A (Stratified)",       "outputs/loso_stratified_baseline",         "stratified"),
    ("Exp B",                    "outputs/loso_improved",                    "stratified"),
    ("Exp C",                    "outputs/cfg3_batch32_reg",                 "stratified"),
    ("Exp D (MT d64)",           "outputs/loso_multitask_uiprmd",            "stratified"),
]

# ---------------------------------------------------------------------------
# Data loading (mirrors train_loso.py)
# ---------------------------------------------------------------------------

def _select_xyz(raw: np.ndarray) -> np.ndarray:
    cols = []
    for j in range(NUM_JOINTS):
        start = j * 4
        cols.extend([start, start + 1, start + 2])
    return raw[:, cols]


def load_pooled(pooled_dir: str):
    x_raw  = pd.read_csv(os.path.join(pooled_dir, "Train_X.csv"),      header=None).values.astype(np.float32)
    y_raw  = pd.read_csv(os.path.join(pooled_dir, "Train_Y.csv"),      header=None).values.squeeze().astype(np.float32)
    sids   = pd.read_csv(os.path.join(pooled_dir, "subject_ids.csv"),  header=None).values.squeeze().astype(np.int32)
    eids   = pd.read_csv(os.path.join(pooled_dir, "exercise_ids.csv"), header=None).values.squeeze().astype(np.int32)
    meta   = pd.read_csv(os.path.join(pooled_dir, "meta.csv"))
    groups = meta["group"].values

    n_samples = y_raw.shape[0]
    xyz = _select_xyz(x_raw)
    xyz_4d = xyz.reshape(n_samples, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)
    return xyz_4d, eids, y_raw, sids, groups

# ---------------------------------------------------------------------------
# Model reconstruction from saved args
# ---------------------------------------------------------------------------

def build_model_from_args(
    args_dict: dict,
    device: torch.device,
    ref_state_dict: dict | None = None,
) -> torch.nn.Module:
    args = SimpleNamespace(**args_dict)
    model_type = getattr(args, "model_type", "transformer")
    multitask  = getattr(args, "multitask", False)

    # Infer num_exercises from the saved embedding weight shape if available
    # (handles old checkpoints trained before UI-PRMD was added)
    num_ex = NUM_EXERCISES
    if ref_state_dict is not None and "exercise_emb.weight" in ref_state_dict:
        num_ex = ref_state_dict["exercise_emb.weight"].shape[0]

    if model_type == "stgcn":
        model = STGCNRegressor(
            seq_len=SEQ_LEN,
            base_channels=getattr(args, "base_channels", 32),
            dropout=args.dropout,
            num_exercises=num_ex,
            multitask=multitask,
        )
    elif model_type == "lstm":
        model = LSTMRegressor(
            seq_len=SEQ_LEN,
            hidden_size=getattr(args, "lstm_hidden", 128),
            num_layers=getattr(args, "lstm_layers", 2),
            dropout=args.dropout,
            num_exercises=num_ex,
            multitask=multitask,
        )
    elif model_type == "tcn":
        model = TCNRegressor(
            seq_len=SEQ_LEN,
            d_model=getattr(args, "d_model", 128),
            num_blocks=getattr(args, "tcn_blocks", 4),
            dropout=args.dropout,
            num_exercises=num_ex,
            multitask=multitask,
        )
    elif model_type == "sct":
        model = SCTRegressor(
            seq_len=SEQ_LEN,
            d_model=getattr(args, "sct_d_model", 64),
            nhead=getattr(args, "sct_heads", 4),
            num_layers=getattr(args, "sct_layers", 2),
            dropout=args.dropout,
            num_exercises=num_ex,
            multitask=multitask,
        )
    else:
        model = build_model(
            model_type=model_type,
            seq_len=SEQ_LEN,
            dropout=args.dropout,
            joint_dim=getattr(args, "joint_dim", 64),
            d_model=getattr(args, "d_model", 128),
            spatial_heads=getattr(args, "spatial_heads", 4),
            spatial_layers=getattr(args, "spatial_layers", 2),
            temporal_heads=getattr(args, "temporal_heads", 4),
            temporal_layers=getattr(args, "temporal_layers", 2),
            num_exercises=num_ex,
            multitask=multitask,
            use_graph_bias=not getattr(args, "no_graph_bias", False),
        )
    return model.to(device)

# ---------------------------------------------------------------------------
# Per-fold inference
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    val_x4d: np.ndarray,
    val_eids: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    multitask: bool = False,
) -> np.ndarray:
    x_t = torch.from_numpy(val_x4d).float()
    e_t = torch.from_numpy(val_eids.astype(np.int64))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_t), batch_size):
            xb = x_t[i : i + batch_size].to(device)
            eb = e_t[i : i + batch_size].to(device)
            out = model(xb, eb)
            if multitask:
                out = out[0]   # TS head only
            preds.append(out.cpu().numpy())
    return np.concatenate(preds).reshape(-1)

# ---------------------------------------------------------------------------
# Main per-experiment function
# ---------------------------------------------------------------------------

def generate_oof(exp_dir: str, split_method: str = "stratified") -> pd.DataFrame | None:
    exp_path = Path(exp_dir)

    # Load args from fold_0 checkpoint (all folds share the same hyperparams)
    ckpt0 = exp_path / "fold_0" / "best_model.pt"
    if not ckpt0.exists():
        print(f"  [SKIP] {exp_dir} — fold_0/best_model.pt not found")
        return None

    ckpt_data = torch.load(str(ckpt0), map_location="cpu", weights_only=False)
    args_dict  = ckpt_data["args"]
    args       = SimpleNamespace(**args_dict)

    # Load dataset from the path recorded in the checkpoint args
    pooled_dir = getattr(args, "pooled_dir", r"D:\Rehabilation\KIMORE_pooled")
    if not os.path.isdir(pooled_dir):
        print(f"  [SKIP] {exp_dir} — pooled_dir not found: {pooled_dir}")
        return None

    xyz_4d, eids, y_raw, sids, groups = load_pooled(pooled_dir)
    n_total = y_raw.shape[0]
    n_folds = getattr(args, "n_folds", 5)

    # Reconstruct identical splits (StratifiedGroupKFold with shuffle=False is deterministic)
    if split_method == "group":
        splitter  = GroupKFold(n_splits=n_folds)
        split_gen = splitter.split(np.arange(n_total), groups=sids)
    else:
        splitter  = StratifiedGroupKFold(n_splits=n_folds)
        split_gen = splitter.split(np.arange(n_total), y=groups, groups=sids)

    splits = list(split_gen)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multitask = getattr(args, "multitask", False)
    T, J, C = SEQ_LEN, NUM_JOINTS, NUM_CHANNELS

    all_oof: list[pd.DataFrame] = []

    for fold, (_, val_idx) in enumerate(splits):
        fold_dir    = exp_path / f"fold_{fold}"
        ckpt_path   = fold_dir / "best_model.pt"
        scaler_path = fold_dir / "scalers.pkl"

        if not ckpt_path.exists() or not scaler_path.exists():
            print(f"  [SKIP] fold_{fold}: missing checkpoint or scalers")
            continue

        # Reconstruct model and load weights (pass state_dict so num_exercises
        # is inferred from checkpoint, handling old pre-UI-PRMD checkpoints)
        ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = build_model_from_args(args_dict, device, ref_state_dict=ckpt["model_state"])
        model.load_state_dict(ckpt["model_state"])

        # Load scalers
        bundle: ScalerBundle = joblib.load(str(scaler_path))

        # Scale val data using training scaler
        n_val        = len(val_idx)
        val_xyz_flat = xyz_4d[val_idx].reshape(n_val * T, J * C)
        val_xyz_sc   = bundle.x_scaler.transform(val_xyz_flat).astype(np.float32)
        val_x4d      = val_xyz_sc.reshape(n_val, T, J, C)

        # Inference
        preds_sc = run_inference(model, val_x4d, eids[val_idx], device,
                                 batch_size=64, multitask=multitask)

        # Inverse-transform to original score space
        preds = bundle.y_scaler.inverse_transform(preds_sc.reshape(-1, 1)).reshape(-1)
        tgts  = y_raw[val_idx]
        rmse  = float(np.sqrt(np.mean((tgts - preds) ** 2)))

        fold_df = pd.DataFrame({
            "subject_id":  sids[val_idx],
            "exercise_id": eids[val_idx],
            "y_true":      tgts,
            "y_pred":      preds,
            "abs_error":   np.abs(tgts - preds),
            "fold":        fold,
        })
        fold_df.to_csv(str(fold_dir / "oof_predictions.csv"), index=False)
        all_oof.append(fold_df)
        print(f"    fold_{fold}: n={n_val}  RMSE={rmse:.4f}  -> {fold_dir}/oof_predictions.csv")

    if not all_oof:
        return None

    combined = pd.concat(all_oof, ignore_index=True)
    out_path = exp_path / "oof_predictions_all.csv"
    combined.to_csv(str(out_path), index=False)
    overall_rmse = float(np.sqrt(np.mean((combined["y_true"] - combined["y_pred"]) ** 2)))
    print(f"    Combined: n={len(combined)}  overall RMSE={overall_rmse:.4f}  -> {out_path}")
    return combined

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir",      default="",
                        help="Single experiment directory to process.")
    parser.add_argument("--split_method", default="stratified",
                        choices=["stratified", "group"],
                        help="CV split method used during training.")
    parser.add_argument("--all",          action="store_true",
                        help="Process all experiments in the registry.")
    args = parser.parse_args()

    if args.all:
        for name, exp_dir, split in EXPERIMENTS:
            print(f"\n[{name}]  {exp_dir}")
            generate_oof(exp_dir, split)
    elif args.exp_dir:
        print(f"\n[{args.exp_dir}]")
        generate_oof(args.exp_dir, args.split_method)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

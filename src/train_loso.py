"""5-fold Leave-One-Subject-Out cross-validation on the pooled KIMORE dataset.

Why this design:
  - Pooled (385 samples) solves data starvation vs per-exercise (77 samples).
  - LOSO prevents subject-level leakage: all 5 recordings from a subject are
    either entirely in train or entirely in val — never split.
  - 5-fold gives stable mean±std metrics across different subject splits,
    which is what paper reviewers and clinical collaborators expect.

Usage:
  python train_loso.py --pooled_dir KIMORE_pooled --out_dir outputs/loso_pooled

Outputs per fold:
  outputs/loso_pooled/fold_0/best_model.pt
  outputs/loso_pooled/fold_0/scalers.pkl
  outputs/loso_pooled/fold_0/train.log
  outputs/loso_pooled/fold_0/training_curves.png

Summary:
  outputs/loso_pooled/loso_results.json
  outputs/loso_pooled/loso_summary.png
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

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from rehab_dataset import NUM_CHANNELS, NUM_JOINTS, ScalerBundle
from models import build_model, count_parameters
from models_stgcn import STGCNRegressor, LSTMRegressor, TCNRegressor, SCTRegressor, count_parameters as count_params_stgcn  # generic parameter counter (works for all models)
from visualize import plot_prediction_scatter, plot_training_curves, plot_residuals

KIMORE_EXERCISES = 5
UIPRMD_EXERCISES = 10
NUM_EXERCISES = KIMORE_EXERCISES + UIPRMD_EXERCISES  # 15 total (KIMORE 0-4, UI-PRMD 5-14)
SEQ_LEN = 100
TOTAL_COLS = NUM_JOINTS * 4   # raw CSV has 25 joints x 4 cols


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PooledSkeletonDataset(Dataset):
    """Pooled dataset: returns (x, exercise_id, y[, po, cf]) with optional augmentation."""

    def __init__(
        self,
        x: np.ndarray,                       # [N, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS]
        exercise_ids: np.ndarray,            # [N] int 0..4
        y: np.ndarray,                       # [N]
        augment: bool = False,
        jitter_std: float = 0.02,
        warp_range: float = 0.15,
        po: np.ndarray | None = None,            # [N] PO sub-score (optional)
        cf: np.ndarray | None = None,            # [N] CF sub-score (optional)
        has_subscores: np.ndarray | None = None, # [N] 1=KIMORE clinical, 0=proxy label
    ) -> None:
        assert x.shape[1:] == (SEQ_LEN, NUM_JOINTS, NUM_CHANNELS), (
            f"x shape error: {x.shape}"
        )
        self._x = x
        self._eid = torch.from_numpy(exercise_ids.astype(np.int64))
        self._y = torch.from_numpy(y.reshape(-1, 1).astype(np.float32))
        self._augment = augment
        self._jitter_std = jitter_std
        self._warp_range = warp_range
        self._po = torch.from_numpy(po.reshape(-1, 1).astype(np.float32)) if po is not None else None
        self._cf = torch.from_numpy(cf.reshape(-1, 1).astype(np.float32)) if cf is not None else None
        # has_subscores: 1 for KIMORE (has PO/CF), 0 for UI-PRMD (proxy labels only)
        if has_subscores is not None:
            self._has_sub = torch.from_numpy(has_subscores.astype(np.float32))
        elif po is not None:
            self._has_sub = torch.ones(self._y.shape[0])
        else:
            self._has_sub = torch.zeros(self._y.shape[0])

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int):
        x_sample = self._x[idx]
        if self._augment:
            if self._warp_range > 0 and np.random.rand() < 0.5:
                x_sample = self._apply_time_warp(x_sample)
            if self._jitter_std > 0 and np.random.rand() < 0.5:
                x_sample = self._apply_jitter(x_sample)

        x_tensor = torch.from_numpy(x_sample.astype(np.float32))
        po_val = self._po[idx] if self._po is not None else torch.zeros(1)
        cf_val = self._cf[idx] if self._cf is not None else torch.zeros(1)
        return x_tensor, self._eid[idx], self._y[idx], po_val, cf_val, self._has_sub[idx]

    def _apply_jitter(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self._jitter_std, size=x.shape).astype(np.float32)
        return x + noise

    def _apply_time_warp(self, x: np.ndarray) -> np.ndarray:
        T, J, C = x.shape
        # warp_range=0.15 → speeds in [0.5, 1.5]; scale factor 3.33 maps [0,1] to that range
        half = self._warp_range * 3.33
        d = np.random.uniform(max(0.1, 1.0 - half), 1.0 + half, size=3)
        times = np.zeros(4, dtype=np.float32)
        times[1:] = np.cumsum(d)
        times = times / times[-1] * (T - 1)
        
        xp = np.array([0, T // 3, 2 * T // 3, T - 1], dtype=np.float32)
        warped_steps = np.interp(np.arange(T), xp, times).astype(np.float32)
        
        x_flat = x.reshape(T, -1)
        x_warped = np.zeros_like(x_flat)
        for col in range(x_flat.shape[1]):
            x_warped[:, col] = np.interp(warped_steps, np.arange(T), x_flat[:, col])
            
        return x_warped.reshape(T, J, C)


def _select_xyz(raw: np.ndarray) -> np.ndarray:
    """Extract x,y,z columns (drop every 4th orientation column)."""
    cols = []
    for j in range(NUM_JOINTS):
        start = j * 4
        cols.extend([start, start + 1, start + 2])
    return raw[:, cols]


def load_pooled(pooled_dir: str):
    """Load pooled CSVs. Returns raw arrays, clinical groups, and optional PO/CF scores."""
    x_raw  = pd.read_csv(os.path.join(pooled_dir, "Train_X.csv"),      header=None).values.astype(np.float32)
    y_raw  = pd.read_csv(os.path.join(pooled_dir, "Train_Y.csv"),      header=None).values.squeeze().astype(np.float32)
    sids   = pd.read_csv(os.path.join(pooled_dir, "subject_ids.csv"),  header=None).values.squeeze().astype(np.int32)
    eids   = pd.read_csv(os.path.join(pooled_dir, "exercise_ids.csv"), header=None).values.squeeze().astype(np.int32)
    meta   = pd.read_csv(os.path.join(pooled_dir, "meta.csv"))
    groups = meta["group"].values

    n_samples = y_raw.shape[0]
    assert x_raw.shape[0] == n_samples * SEQ_LEN, (
        f"Train_X rows {x_raw.shape[0]} != {n_samples}*{SEQ_LEN}"
    )

    xyz = _select_xyz(x_raw)                              # [N*T, J*C]
    xyz_4d = xyz.reshape(n_samples, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)

    # Load sub-scores for multi-task learning if available
    po_path = os.path.join(pooled_dir, "Train_PO.csv")
    cf_path = os.path.join(pooled_dir, "Train_CF.csv")
    if os.path.exists(po_path) and os.path.exists(cf_path):
        po_raw = pd.read_csv(po_path, header=None).values.squeeze().astype(np.float32)
        cf_raw = pd.read_csv(cf_path, header=None).values.squeeze().astype(np.float32)
    else:
        po_raw = None
        cf_raw = None

    return xyz_4d, eids, y_raw, sids, groups, po_raw, cf_raw


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _get_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# One training run (single fold)
# ---------------------------------------------------------------------------

def _load_uiprmd(uiprmd_dir: str):
    """Load preprocessed UI-PRMD data. Returns None tuple if dir not given."""
    if not uiprmd_dir or not os.path.isdir(uiprmd_dir):
        return None, None, None, None
    x_raw = pd.read_csv(os.path.join(uiprmd_dir, "Train_X.csv"),     header=None).values.astype(np.float32)
    y_raw = pd.read_csv(os.path.join(uiprmd_dir, "Train_Y.csv"),     header=None).values.squeeze().astype(np.float32)
    eids  = pd.read_csv(os.path.join(uiprmd_dir, "exercise_ids.csv"),header=None).values.squeeze().astype(np.int32)
    n     = y_raw.shape[0]
    assert x_raw.shape[0] == n * SEQ_LEN
    xyz = _select_xyz(x_raw)
    xyz_4d = xyz.reshape(n, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)
    return xyz_4d, eids, y_raw, n


def _load_pretrained_encoder(model, ckpt_path: str, logger) -> None:
    """Load an SSL-pretrained encoder (src/ssl) into a fresh regressor.

    The checkpoint stores encoder-only weights (regression head stripped), so
    strict=False loads the shared backbone params and leaves the head fresh.
    """
    blob = torch.load(ckpt_path, map_location="cpu")
    state = blob.get("encoder_state", blob.get("model_state", blob))
    # Keep only keys whose shape matches the current model (strict=False still
    # RAISES on shape mismatch, so filter first -> robust to dim differences).
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    dropped = [k for k in state if k not in compatible]
    res = model.load_state_dict(compatible, strict=False)
    logger.info(
        f"init_ckpt={ckpt_path} | loaded {len(compatible)} encoder tensors | "
        f"fresh(head) missing={len(res.missing_keys)} | shape/name-dropped={len(dropped)}"
    )
    if dropped:
        logger.warning(f"init_ckpt: dropped {len(dropped)} incompatible keys (e.g. {dropped[:3]})")
    if len(compatible) == 0:
        logger.warning("init_ckpt: NO encoder tensors matched -- check d_model / model_type.")


def _freeze_encoder(model, logger) -> None:
    """Freeze all params except the regression head(s) -> linear-probe condition."""
    frozen = trainable = 0
    for name, p in model.named_parameters():
        if name.startswith("head") or name.startswith("exercise_emb"):
            trainable += p.numel()
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    logger.info(f"freeze_encoder | frozen={frozen} trainable(head)={trainable}")


def _metrics_from_oof(fold: int, oof_path: Path) -> dict:
    """Reconstruct a fold's metrics dict from a saved oof_predictions.csv.

    Used by --resume to reuse folds completed before the metrics.json marker
    existed (e.g. an earlier run interrupted by power loss). Metric values are
    deterministic from the saved predictions; 'epoch' is unknown so set to -1.
    """
    df = pd.read_csv(str(oof_path))
    yt = df["y_true"].values.astype(float)
    yp = df["y_pred"].values.astype(float)
    eids = df["exercise_id"].values
    n = len(yt)
    rmse = float(np.sqrt(mean_squared_error(yt, yp))) if n >= 1 else float("nan")
    mae  = float(mean_absolute_error(yt, yp)) if n >= 1 else float("nan")
    r2   = float(r2_score(yt, yp)) if n >= 2 else float("nan")
    pear = float(pearsonr(yt, yp)[0]) if n >= 2 and yt.std() > 0 and yp.std() > 0 else float("nan")
    spear = float(spearmanr(yt, yp)[0]) if n >= 2 else float("nan")
    per_ex = {}
    for e in np.unique(eids):
        m = eids == e
        if m.sum() >= 2:
            per_ex[f"Ex{int(e)}"] = {
                "n":    int(m.sum()),
                "rmse": float(np.sqrt(mean_squared_error(yt[m], yp[m]))),
                "mae":  float(mean_absolute_error(yt[m], yp[m])),
                "r2":   float(r2_score(yt[m], yp[m])),
            }
    return {"fold": fold, "epoch": -1, "rmse": rmse, "mae": mae, "r2": r2,
            "pearson": pear, "spearman": spear, "per_exercise": per_ex}


def train_one_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    xyz_4d: np.ndarray,
    eids: np.ndarray,
    y_raw: np.ndarray,
    args,
    device: torch.device,
    sids: np.ndarray | None = None,        # subject IDs [N] for OOF CSV
    po_raw: np.ndarray | None = None,
    cf_raw: np.ndarray | None = None,
    extra_xyz: np.ndarray | None = None,   # UI-PRMD xyz [M, T, J, C]
    extra_eids: np.ndarray | None = None,  # UI-PRMD exercise IDs [M]
    extra_y: np.ndarray | None = None,     # UI-PRMD quality scores [M]
) -> dict:
    out_dir = Path(args.out_dir) / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _get_logger(str(out_dir / "train.log"))

    multitask   = getattr(args, "multitask", False) and (po_raw is not None)
    has_extra   = extra_xyz is not None
    T, J, C     = SEQ_LEN, NUM_JOINTS, NUM_CHANNELS

    # ── Scale — fit on KIMORE train only, then apply to KIMORE val + UI-PRMD ──
    n_kimore_train = len(train_idx)
    n_val          = len(val_idx)

    train_xyz_flat = xyz_4d[train_idx].reshape(n_kimore_train * T, J * C)
    val_xyz_flat   = xyz_4d[val_idx].reshape(n_val * T, J * C)

    x_scaler  = StandardScaler()
    y_scaler  = StandardScaler()
    po_scaler = StandardScaler() if multitask else None
    cf_scaler = StandardScaler() if multitask else None

    train_xyz_sc = x_scaler.fit_transform(train_xyz_flat).astype(np.float32)
    val_xyz_sc   = x_scaler.transform(val_xyz_flat).astype(np.float32)

    train_y_sc = y_scaler.fit_transform(y_raw[train_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    val_y_sc   = y_scaler.transform(y_raw[val_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_x4d = train_xyz_sc.reshape(n_kimore_train, T, J, C)
    val_x4d   = val_xyz_sc.reshape(n_val, T, J, C)

    train_po_sc = train_cf_sc = val_po_sc = val_cf_sc = None
    if multitask:
        train_po_sc = po_scaler.fit_transform(po_raw[train_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
        val_po_sc   = po_scaler.transform(po_raw[val_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
        train_cf_sc = cf_scaler.fit_transform(cf_raw[train_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
        val_cf_sc   = cf_scaler.transform(cf_raw[val_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    kimore_has_sub = np.ones(n_kimore_train, dtype=np.float32)

    bundle = ScalerBundle(
        x_scaler=x_scaler, y_scaler=y_scaler,
        po_scaler=po_scaler if multitask else None,
        cf_scaler=cf_scaler if multitask else None,
    )
    joblib.dump(bundle, str(out_dir / "scalers.pkl"))

    # ── Combine KIMORE train with UI-PRMD (always train, never val) ─────────
    if has_extra:
        n_extra = extra_xyz.shape[0]
        extra_xyz_flat = extra_xyz.reshape(n_extra * T, J * C)
        extra_xyz_sc   = x_scaler.transform(extra_xyz_flat).astype(np.float32)
        extra_x4d      = extra_xyz_sc.reshape(n_extra, T, J, C)
        extra_y_sc     = y_scaler.transform(extra_y.reshape(-1, 1)).reshape(-1).astype(np.float32)
        extra_has_sub  = np.zeros(n_extra, dtype=np.float32)  # no PO/CF for UI-PRMD

        all_train_x4d  = np.concatenate([train_x4d, extra_x4d], axis=0)
        all_train_eids = np.concatenate([eids[train_idx], extra_eids], axis=0)
        all_train_y    = np.concatenate([train_y_sc, extra_y_sc], axis=0)
        all_has_sub    = np.concatenate([kimore_has_sub, extra_has_sub], axis=0)
        all_train_po   = np.concatenate([train_po_sc, np.zeros(n_extra, dtype=np.float32)], axis=0) if multitask else None
        all_train_cf   = np.concatenate([train_cf_sc, np.zeros(n_extra, dtype=np.float32)], axis=0) if multitask else None
        n_train_total  = n_kimore_train + n_extra
    else:
        all_train_x4d  = train_x4d
        all_train_eids = eids[train_idx]
        all_train_y    = train_y_sc
        all_has_sub    = kimore_has_sub
        all_train_po   = train_po_sc
        all_train_cf   = train_cf_sc
        n_train_total  = n_kimore_train

    train_ds = PooledSkeletonDataset(
        all_train_x4d, all_train_eids, all_train_y,
        augment=args.augment, jitter_std=args.jitter_std, warp_range=args.warp_range,
        po=all_train_po, cf=all_train_cf, has_subscores=all_has_sub,
    )
    val_ds = PooledSkeletonDataset(
        val_x4d, eids[val_idx], val_y_sc,
        augment=False, po=val_po_sc, cf=val_cf_sc,
        has_subscores=np.ones(n_val, dtype=np.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model_type = getattr(args, "model_type", "transformer")

    if model_type == "stgcn":
        model = STGCNRegressor(
            seq_len=SEQ_LEN,
            base_channels=getattr(args, "base_channels", 32),
            dropout=args.dropout,
            num_exercises=NUM_EXERCISES,
            multitask=multitask,
        ).to(device)
        n_params = count_params_stgcn(model)
    elif model_type == "lstm":
        model = LSTMRegressor(
            seq_len=SEQ_LEN,
            hidden_size=getattr(args, "lstm_hidden", 128),
            num_layers=getattr(args, "lstm_layers", 2),
            dropout=args.dropout,
            num_exercises=NUM_EXERCISES,
            multitask=multitask,
        ).to(device)
        n_params = count_params_stgcn(model)
    elif model_type == "tcn":
        model = TCNRegressor(
            seq_len=SEQ_LEN,
            d_model=getattr(args, "d_model", 128),
            num_blocks=getattr(args, "tcn_blocks", 4),
            dropout=args.dropout,
            num_exercises=NUM_EXERCISES,
            multitask=multitask,
        ).to(device)
        n_params = count_params_stgcn(model)
    elif model_type == "sct":
        model = SCTRegressor(
            seq_len=SEQ_LEN,
            d_model=getattr(args, "sct_d_model", 64),
            nhead=getattr(args, "sct_heads", 4),
            num_layers=getattr(args, "sct_layers", 2),
            dropout=args.dropout,
            num_exercises=NUM_EXERCISES,
            multitask=multitask,
        ).to(device)
        n_params = count_params_stgcn(model)
    else:
        model = build_model(
            model_type=model_type,   # "transformer" or "graph_transformer"
            seq_len=SEQ_LEN,
            dropout=args.dropout,
            joint_dim=args.joint_dim,
            d_model=args.d_model,
            spatial_heads=args.spatial_heads,
            spatial_layers=args.spatial_layers,
            temporal_heads=args.temporal_heads,
            temporal_layers=args.temporal_layers,
            num_exercises=NUM_EXERCISES,
            multitask=multitask,
            use_graph_bias=not getattr(args, "no_graph_bias", False),
        ).to(device)
        n_params = count_parameters(model)

    # --- Paper-2 SSL hook: initialize from a pretrained encoder (optional) ---
    if getattr(args, "init_ckpt", ""):
        _load_pretrained_encoder(model, args.init_ckpt, logger)
        if getattr(args, "freeze_encoder", False):
            _freeze_encoder(model, logger)

    logger.info(
        f"Fold {fold} | model={model_type} | kimore_train={n_kimore_train} "
        f"extra={n_train_total - n_kimore_train} val={n_val} | "
        f"params={n_params:,} | multitask={multitask}"
    )

    opt        = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn    = nn.HuberLoss(reduction="mean", delta=args.huber_delta)
    loss_fn_nr = nn.HuberLoss(reduction="none", delta=args.huber_delta)  # for masked sub-score loss
    amp_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    patience   = getattr(args, "patience", 20)
    aux_weight = getattr(args, "aux_weight", 0.3)

    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_mae": [], "val_r2": []}
    best_rmse, best_metrics, no_improve = float("inf"), {}, 0

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"F{fold} E{epoch}/{args.epochs}", leave=False):
            xb, eid_b, yb, pob, cfb, has_sub_b = batch
            xb        = xb.to(device, non_blocking=True).float()
            eid_b     = eid_b.to(device, non_blocking=True)
            yb        = yb.to(device, non_blocking=True).float()
            pob       = pob.to(device, non_blocking=True).float()
            cfb       = cfb.to(device, non_blocking=True).float()
            has_sub_b = has_sub_b.to(device, non_blocking=True).float()  # [B]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if multitask:
                    out_ts, out_po, out_cf = model(xb, eid_b)
                    ts_loss = loss_fn(out_ts, yb)
                    # PO/CF loss masked to KIMORE-only samples (has_sub_b == 1)
                    n_sub = has_sub_b.sum().clamp(min=1)
                    po_loss = (loss_fn_nr(out_po, pob).squeeze(1) * has_sub_b).sum() / n_sub
                    cf_loss = (loss_fn_nr(out_cf, cfb).squeeze(1) * has_sub_b).sum() / n_sub
                    loss = ts_loss + aux_weight * po_loss + aux_weight * cf_loss
                else:
                    out_ts = model(xb, eid_b)
                    loss   = loss_fn(out_ts, yb)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(opt)
            amp_scaler.update()
            total += loss.item() * xb.size(0)
            n     += xb.size(0)
        sched.step()
        train_loss = total / max(n, 1)

        # ── Validate (KIMORE only — always has subscores) ─────────────────
        model.eval()
        preds_sc, tgts_sc, val_eids_all = [], [], []
        val_total, val_n  = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                xb, eid_b, yb, pob, cfb, has_sub_b = batch
                xb    = xb.to(device).float()
                eid_b = eid_b.to(device)
                yb    = yb.to(device).float()
                pob   = pob.to(device).float()
                cfb   = cfb.to(device).float()
                if multitask:
                    out_ts, out_po, out_cf = model(xb, eid_b)
                    val_total += (loss_fn(out_ts, yb) + aux_weight * loss_fn(out_po, pob) + aux_weight * loss_fn(out_cf, cfb)).item() * xb.size(0)
                else:
                    out_ts = model(xb, eid_b)
                    val_total += loss_fn(out_ts, yb).item() * xb.size(0)
                val_n += xb.size(0)
                preds_sc.append(out_ts.cpu().numpy())
                tgts_sc.append(yb.cpu().numpy())
                val_eids_all.append(eid_b.cpu().numpy())

        val_loss     = val_total / max(val_n, 1)
        preds_sc     = np.concatenate(preds_sc).reshape(-1)
        tgts_sc      = np.concatenate(tgts_sc).reshape(-1)
        val_eids_cat = np.concatenate(val_eids_all).reshape(-1)

        # Inverse-transform for real-scale metrics (TS only — the primary target)
        preds = y_scaler.inverse_transform(preds_sc.reshape(-1, 1)).reshape(-1)
        tgts  = y_scaler.inverse_transform(tgts_sc.reshape(-1, 1)).reshape(-1)
        rmse    = float(np.sqrt(mean_squared_error(tgts, preds)))
        mae     = float(mean_absolute_error(tgts, preds))
        r2      = float(r2_score(tgts, preds)) if len(tgts) > 1 else float("nan")
        pearson = float(pearsonr(tgts, preds)[0]) if len(tgts) > 1 else float("nan")
        spearman = float(spearmanr(tgts, preds)[0]) if len(tgts) > 1 else float("nan")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(rmse)
        history["val_mae"].append(mae)
        history["val_r2"].append(r2)

        logger.info(json.dumps({"fold": fold, "epoch": epoch,
                                "train_loss": round(train_loss, 5),
                                "val_loss":   round(val_loss, 5),
                                "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4),
                                "pearson": round(pearson, 4), "spearman": round(spearman, 4)}))

        if rmse < best_rmse:
            best_rmse  = rmse
            no_improve = 0

            # Per-exercise breakdown at this (new best) epoch
            per_ex: dict[str, dict] = {}
            for eid_val in np.unique(val_eids_cat):
                mask = val_eids_cat == eid_val
                if mask.sum() < 2:
                    continue
                ex_key = f"Ex{eid_val}"
                per_ex[ex_key] = {
                    "n":    int(mask.sum()),
                    "rmse": float(np.sqrt(mean_squared_error(tgts[mask], preds[mask]))),
                    "mae":  float(mean_absolute_error(tgts[mask], preds[mask])),
                    "r2":   float(r2_score(tgts[mask], preds[mask])),
                }

            best_metrics = {
                "fold": fold, "epoch": epoch,
                "rmse": rmse, "mae": mae, "r2": r2,
                "pearson": pearson, "spearman": spearman,
                "per_exercise": per_ex,
            }
            torch.save({"model_state": model.state_dict(), "args": vars(args),
                        "epoch": epoch, "val_rmse": best_rmse},
                       str(out_dir / "best_model.pt"))

            # Save OOF predictions for sample-level statistical tests
            val_sids = sids[val_idx] if sids is not None else np.arange(len(val_idx))
            oof_df = pd.DataFrame({
                "subject_id":  val_sids,
                "exercise_id": val_eids_cat,
                "y_true":      tgts,
                "y_pred":      preds,
                "abs_error":   np.abs(tgts - preds),
                "fold":        fold,
            })
            oof_df.to_csv(str(out_dir / "oof_predictions.csv"), index=False)
            ex_labels = [f"Ex{eid}" for eid in val_eids_cat]
            plot_prediction_scatter(tgts, preds, str(out_dir / "prediction_scatter.png"),
                                    title=f"Fold {fold} epoch {epoch} RMSE={best_rmse:.3f}")
            plot_residuals(tgts, preds, str(out_dir / "residuals.png"),
                           group_labels=ex_labels)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    plot_training_curves(history, str(out_dir / "training_curves.png"))
    np.save(str(out_dir / "history.npy"), history)
    logger.info(f"Fold {fold} done. Best RMSE={best_rmse:.4f}")
    return best_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled_dir", default=r"D:\Rehabilation\KIMORE_pooled")
    parser.add_argument("--out_dir",    default=r"D:\Rehabilation\outputs\loso_pooled")
    parser.add_argument("--n_folds",    type=int,   default=5)
    parser.add_argument("--loso", action="store_true",
                        help="True leave-one-subject-out (LeaveOneGroupOut): one fold per subject, "
                             "ignores --n_folds. Not stratified per fold (each fold is one subject).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip folds that already finished (metrics.json marker, or reconstruct "
                             "from an existing oof_predictions.csv) and continue at the first "
                             "incomplete fold. Survives power loss / load-shedding — at most one "
                             "in-progress fold is ever lost.")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--huber_delta", type=float, default=0.1)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--joint_dim",  type=int,   default=64)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--spatial_heads",  type=int, default=4)
    parser.add_argument("--spatial_layers", type=int, default=2)
    parser.add_argument("--temporal_heads",  type=int, default=4)
    parser.add_argument("--temporal_layers", type=int, default=2)
    parser.add_argument("--seed",       type=int,   default=145)
    parser.add_argument("--augment",    action="store_true", help="Enable data augmentation for training.")
    parser.add_argument("--jitter_std", type=float, default=0.02, help="Standard deviation of Gaussian jitter noise.")
    parser.add_argument("--warp_range", type=float, default=0.15, help="Time warping intensity range [0,1]; default 0.15 -> speeds in [0.5, 1.5].")
    parser.add_argument("--multitask",  action="store_true", help="Also predict PO and CF sub-scores as auxiliary losses (requires Train_PO/CF.csv).")
    parser.add_argument("--patience",   type=int, default=20,  help="Early stopping: epochs without improvement before stopping.")
    parser.add_argument("--uiprmd_dir",  default="",    help="Path to preprocessed UI-PRMD dataset directory (optional).")
    parser.add_argument("--aux_weight",  type=float, default=0.3, help="Loss weight scale for auxiliary PO/CF heads in multitask learning.")
    # Model type selection
    parser.add_argument("--model_type", default="transformer",
                        choices=["transformer", "graph_transformer", "stgcn", "lstm", "tcn", "sct"],
                        help="Model architecture: transformer, graph_transformer, stgcn, lstm, tcn, sct.")
    # Paper-2 SSL fine-tuning hooks (see src/ssl/):
    parser.add_argument("--init_ckpt", default="",
                        help="Path to an SSL-pretrained encoder checkpoint (src/ssl/pretrain.py). "
                             "Loaded with strict=False; the regression head stays freshly initialized.")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze the encoder and train only the regression head (linear-probe condition).")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="ST-GCN base channel width (doubles each block). Only used with --model_type stgcn.")
    parser.add_argument("--lstm_hidden", type=int, default=128,
                        help="BiLSTM hidden size per direction. Only used with --model_type lstm.")
    parser.add_argument("--lstm_layers", type=int, default=2,
                        help="Number of BiLSTM layers. Only used with --model_type lstm.")
    parser.add_argument("--tcn_blocks", type=int, default=4,
                        help="Number of TCN dilated causal conv blocks (dilation 1,2,4,8,...). Only used with --model_type tcn.")
    parser.add_argument("--sct_d_model", type=int, default=64,
                        help="SCT token embedding dimension. Only used with --model_type sct.")
    parser.add_argument("--sct_heads", type=int, default=4,
                        help="SCT attention heads. Only used with --model_type sct.")
    parser.add_argument("--sct_layers", type=int, default=2,
                        help="SCT transformer encoder layers. Only used with --model_type sct.")
    parser.add_argument("--no_graph_bias", action="store_true",
                        help="Disable bone-distance attention bias in GraphTransformer (ablation).")
    args = parser.parse_args()

    seed_everything(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    xyz_4d, eids, y_raw, sids, groups, po_raw, cf_raw = load_pooled(args.pooled_dir)
    n_total = y_raw.shape[0]
    print(f"Pooled dataset: {n_total} samples | "
          f"subjects: {len(np.unique(sids))} | exercises: {NUM_EXERCISES}")
    print(f"Score range: [{y_raw.min():.2f}, {y_raw.max():.2f}]  mean={y_raw.mean():.2f}")
    if po_raw is not None:
        print(f"PO range: [{po_raw.min():.2f}, {po_raw.max():.2f}]  "
              f"CF range: [{cf_raw.min():.2f}, {cf_raw.max():.2f}]")
    if args.multitask and po_raw is None:
        print("[WARN] --multitask requested but Train_PO.csv / Train_CF.csv not found. "
              "Re-run prepare_kimore.py then pool_exercises.py first. Falling back to single-task.")

    # Load extra UI-PRMD dataset if directory is provided
    extra_xyz, extra_eids, extra_y, n_extra = _load_uiprmd(args.uiprmd_dir)
    if extra_xyz is not None:
        print(f"Loaded extra UI-PRMD dataset: {n_extra} samples | exercises: 5-14")

    # Fold splitter: true leave-one-subject-out (one fold/subject) or stratified group k-fold.
    if getattr(args, "loso", False):
        splitter = LeaveOneGroupOut()
        fold_iter = splitter.split(np.arange(n_total), y=groups, groups=sids)
        print(f"Fold protocol: LeaveOneGroupOut (true LOSO) -> {len(np.unique(sids))} folds")
    else:
        splitter = StratifiedGroupKFold(n_splits=args.n_folds)
        fold_iter = splitter.split(np.arange(n_total), y=groups, groups=sids)
        print(f"Fold protocol: StratifiedGroupKFold, n_splits={args.n_folds}")
    fold_results = []
    t0 = time.perf_counter()

    resume = getattr(args, "resume", False)
    for fold, (train_idx, val_idx) in enumerate(fold_iter):
        fold_dir     = Path(args.out_dir) / f"fold_{fold}"
        metrics_path = fold_dir / "metrics.json"
        oof_path     = fold_dir / "oof_predictions.csv"

        # ── Resume: reuse a fold that already finished ──────────────────────
        if resume and metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                fold_results.append(metrics)
                print(f"[resume] FOLD {fold}: loaded cached metrics.json (skip)")
                continue
            except (json.JSONDecodeError, OSError):
                print(f"[resume] FOLD {fold}: metrics.json unreadable — retraining")
        if resume and oof_path.exists():
            try:
                metrics = _metrics_from_oof(fold, oof_path)
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f)
                fold_results.append(metrics)
                print(f"[resume] FOLD {fold}: reconstructed from existing oof (skip)")
                continue
            except Exception as e:
                print(f"[resume] FOLD {fold}: oof reconstruction failed ({e}) — retraining")

        print(f"\n{'='*60}")
        print(f"  FOLD {fold}  |  train={len(train_idx)}  val={len(val_idx)}  "
              f"val_subjects={len(np.unique(sids[val_idx]))}")
        print(f"{'='*60}")
        metrics = train_one_fold(fold, train_idx, val_idx,
                                 xyz_4d, eids, y_raw, args, device,
                                 sids=sids,
                                 po_raw=po_raw, cf_raw=cf_raw,
                                 extra_xyz=extra_xyz, extra_eids=extra_eids, extra_y=extra_y)
        # Checkpoint marker: written only after the fold fully completes.
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        fold_results.append(metrics)
        print(f"  Best -> RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    rmses     = [r["rmse"]     for r in fold_results]
    maes      = [r["mae"]      for r in fold_results]
    r2s       = [r["r2"]       for r in fold_results]
    pearsons  = [r.get("pearson",  float("nan")) for r in fold_results]
    spearmans = [r.get("spearman", float("nan")) for r in fold_results]

    # Aggregate per-exercise metrics across folds
    per_ex_agg: dict[str, dict[str, list]] = {}
    for fr in fold_results:
        for ex_key, ex_m in fr.get("per_exercise", {}).items():
            if ex_key not in per_ex_agg:
                per_ex_agg[ex_key] = {"rmse": [], "mae": [], "r2": [], "n": []}
            per_ex_agg[ex_key]["rmse"].append(ex_m["rmse"])
            per_ex_agg[ex_key]["mae"].append(ex_m["mae"])
            per_ex_agg[ex_key]["r2"].append(ex_m["r2"])
            per_ex_agg[ex_key]["n"].append(ex_m["n"])
    per_ex_summary = {
        k: {
            "mean_rmse": float(np.mean(v["rmse"])), "std_rmse": float(np.std(v["rmse"])),
            "mean_r2":   float(np.mean(v["r2"])),   "total_n":  int(sum(v["n"])),
        }
        for k, v in per_ex_agg.items()
    }

    summary = {
        "model_type":   getattr(args, "model_type", "transformer"),
        "folds":        fold_results,
        "mean_rmse":    float(np.mean(rmses)),    "std_rmse":    float(np.std(rmses)),
        "mean_mae":     float(np.mean(maes)),     "std_mae":     float(np.std(maes)),
        "mean_r2":      float(np.mean(r2s)),      "std_r2":      float(np.std(r2s)),
        "mean_pearson": float(np.nanmean(pearsons)), "std_pearson": float(np.nanstd(pearsons)),
        "mean_spearman":float(np.nanmean(spearmans)),"std_spearman":float(np.nanstd(spearmans)),
        "r2_min": float(np.min(r2s)), "r2_max": float(np.max(r2s)),
        "per_exercise": per_ex_summary,
        "elapsed_s": round(elapsed, 1),
    }
    with open(os.path.join(args.out_dir, "loso_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  {len(fold_results)}-FOLD LOSO SUMMARY  ({elapsed:.0f}s)  [{getattr(args,'model_type','transformer').upper()}]")
    print(f"{'='*60}")
    print(f"  RMSE     : {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"  MAE      : {np.mean(maes):.4f}  ± {np.std(maes):.4f}")
    print(f"  R²       : {np.mean(r2s):.4f}  ± {np.std(r2s):.4f}  (range [{np.min(r2s):.3f}, {np.max(r2s):.3f}])")
    print(f"  Pearson  : {np.nanmean(pearsons):.4f}  ± {np.nanstd(pearsons):.4f}")
    print(f"  Spearman : {np.nanmean(spearmans):.4f}  ± {np.nanstd(spearmans):.4f}")
    if per_ex_summary:
        print(f"\n  Per-exercise breakdown:")
        for ex_key in sorted(per_ex_summary):
            m = per_ex_summary[ex_key]
            print(f"    {ex_key}: RMSE={m['mean_rmse']:.3f}±{m['std_rmse']:.3f}  R²={m['mean_r2']:.3f}  n={m['total_n']}")
    print(f"  Results -> {args.out_dir}/loso_results.json")


if __name__ == "__main__":
    main()

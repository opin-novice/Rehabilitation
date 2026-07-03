"""Generic SSL pretraining loop (Layer 2). Task-agnostic: consumes any PretextTask.

Saves an ENCODER-ONLY checkpoint (regression-head params stripped) plus a
provenance sidecar, so it loads cleanly into train_loso.py via --init_ckpt.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models_stgcn import build_encoder, encoder_state_dict  # noqa: E402

from .config import PretrainCfg, provenance  # noqa: E402
from .pretext import build_pretext  # noqa: E402


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretrain(
    x_pool: np.ndarray,
    cfg: PretrainCfg,
    out_dir: str,
    device: Optional[str] = None,
    log_every: int = 10,
) -> str:
    """Pretrain an encoder on unlabeled `x_pool` (N, T, J, C). Returns ckpt path."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _seed(cfg.seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    encoder = build_encoder(
        cfg.model_type, seq_len=cfg.seq_len, d_model=cfg.d_model, dropout=cfg.dropout,
    ).to(device)

    task_kw = {}
    if cfg.pretext == "contrastive":
        task_kw = dict(temperature=cfg.temperature, proj_dim=cfg.proj_dim,
                       aug_exclude=cfg.aug_exclude)
    else:
        task_kw = dict(mask_ratio=cfg.mask_ratio, mask_type=cfg.mask_type,
                       seq_len=cfg.seq_len, num_joints=cfg.num_joints,
                       num_channels=cfg.num_channels)
    task = build_pretext(cfg.pretext, **task_kw)
    head = task.build_head(encoder).to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)

    ds = TensorDataset(torch.from_numpy(x_pool.astype(np.float32)))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    encoder.train(); head.train()
    history = []
    for epoch in range(cfg.epochs):
        running, nb = 0.0, 0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            loss = task.loss(encoder, head, xb)
            loss.backward()
            opt.step()
            running += float(loss.item()); nb += 1
        sched.step()
        avg = running / max(nb, 1)
        history.append(avg)
        if epoch % log_every == 0 or epoch == cfg.epochs - 1:
            print(f"[pretrain:{cfg.pretext}] epoch {epoch:3d}  loss={avg:.4f}")

    ckpt_path = out / f"{cfg.pretext}_encoder.pt"
    torch.save(
        {
            "encoder_state": encoder_state_dict(encoder),
            "model_type": cfg.model_type,
            "out_dim": encoder.out_dim,
            "provenance": provenance(cfg, {"final_loss": history[-1] if history else None,
                                          "n_pool": int(x_pool.shape[0])}),
        },
        str(ckpt_path),
    )
    (out / f"{cfg.pretext}_history.json").write_text(json.dumps(history))
    print(f"[pretrain] saved encoder -> {ckpt_path}")
    return str(ckpt_path)

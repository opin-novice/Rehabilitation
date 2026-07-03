"""Visualization utilities for rehabilitation scoring experiments."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from constants import SKELETON_EDGES


def plot_training_curves(history: dict[str, list[float]], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber loss (scaled)")
    axes[0].set_title("Training and validation loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_rmse"], label="RMSE", color="tab:red")
    axes[1].plot(epochs, history["val_mae"], label="MAE", color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (original scale)")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str = "Prediction vs ground truth",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="none", s=30)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Ground-truth score")
    ax.set_ylabel("Predicted score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_skeleton_frame(
    frame_xyz: np.ndarray,
    out_path: str,
    title: str = "Skeleton frame",
) -> None:
    """Plot a single skeleton frame.

    Args:
        frame_xyz: float array, shape (25, 3) — one frame's joint positions.
    """
    frame_xyz = np.asarray(frame_xyz)
    assert frame_xyz.shape == (25, 3), (
        f"frame_xyz must be (25, 3), got {frame_xyz.shape}"
    )

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(frame_xyz[:, 0], frame_xyz[:, 1], frame_xyz[:, 2], s=20, c="steelblue")

    for a, b in SKELETON_EDGES:
        ax.plot(
            [frame_xyz[a, 0], frame_xyz[b, 0]],
            [frame_xyz[a, 1], frame_xyz[b, 1]],
            [frame_xyz[a, 2], frame_xyz[b, 2]],
            color="gray", linewidth=1,
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    group_labels: list[str] | None = None,
) -> None:
    """Residual scatter plot, optionally coloured by exercise/group label.

    Args:
        group_labels: string label per sample (e.g. "Ex1", "Ex2", clinical group).
                      When provided, each unique label gets a distinct colour and
                      a legend entry so systematic per-group bias is visible.
    """
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(9, 4))

    if group_labels is not None:
        unique = sorted(set(group_labels))
        cmap = plt.get_cmap("tab10")
        for i, grp in enumerate(unique):
            mask = np.array(group_labels) == grp
            ax.scatter(y_true[mask], residuals[mask],
                       alpha=0.65, s=28, edgecolors="none",
                       color=cmap(i % 10), label=grp)
        ax.legend(title="Exercise", fontsize=7, title_fontsize=8,
                  loc="upper right", framealpha=0.7)
    else:
        ax.scatter(y_true, residuals, alpha=0.6, s=25, edgecolors="none", color="steelblue")

    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Ground-truth score")
    ax.set_ylabel("Residual (pred − true)")
    ax.set_title("Residual plot by exercise type")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

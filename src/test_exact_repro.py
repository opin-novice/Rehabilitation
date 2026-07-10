"""EXACT reproduction of original repo pipeline (including data leakage).

Original repo fits StandardScaler on ALL data (train+test) before splitting.
This is technically data leakage but matches the paper's reported results.
"""
import sys, time, json, os
sys.path.insert(0, "D:/Rehabilation/src")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models_curvenet import PointCloudTransformerRegressor
from train_reproduce import extract_body_parts, reshape_to_sequences, performance_metrics, seed_everything

# Constants
BODY_PARTS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]
NUM_JOINTS = 25
NUM_CHANNELS = 3
SEQ_LEN = 100

def load_kimore_exact(exercise_dir, test_size=0.2, seed=420):
    """EXACT match to original repo's Data_Loader + build_dataloaders."""
    train_x_raw = pd.read_csv(os.path.join(exercise_dir, "Train_X.csv"), header=None).values
    train_y_raw = pd.read_csv(os.path.join(exercise_dir, "Train_Y.csv"), header=None).values
    n_samples = train_x_raw.shape[0] // SEQ_LEN

    # Extract body parts (same as original)
    X = extract_body_parts(train_x_raw)
    y = train_y_raw.reshape(-1, 1)

    # Fit scalers on ALL data (matches original - data leakage)
    sc1 = StandardScaler()
    X_scaled = sc1.fit_transform(X)

    sc2 = StandardScaler()
    y_scaled = sc2.fit_transform(y)

    # Reshape X to sequences
    X_4d = reshape_to_sequences(X_scaled, n_samples, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)

    # Split (same as original)
    X_train, X_test, y_train, y_test = train_test_split(
        X_4d, y_scaled, test_size=test_size, random_state=seed
    )

    print(f"  X train: {X_train.shape}, test: {X_test.shape}")
    print(f"  Y train: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"  Y test:  [{y_test.min():.4f}, {y_test.max():.4f}]")
    print(f"  Scaler: mean={sc2.mean_[0]:.4f}, std={sc2.scale_[0]:.4f}")

    return X_train, X_test, y_train, y_test, sc1, sc2

def evaluate_mad_scaled(model, loader, device):
    """Evaluate on SCALED values (for model selection during training)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).float()
            pred = model(x)
            preds.extend(pred.cpu().numpy().flatten())
            trues.extend(y.cpu().numpy().flatten())
    return performance_metrics(trues, preds)[0]

def evaluate_original_scale(model, loader, device, scaler_y):
    """Evaluate on ORIGINAL scale (matches eval.py)."""
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
    preds = scaler_y.inverse_transform(preds)
    trues = scaler_y.inverse_transform(trues)
    return performance_metrics(trues, preds)[0]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed_everything(420)

    print("\nLoading data (EXACT original pipeline - scaler on ALL data)...")
    X_train, X_test, y_train, y_test, sc_x, sc_y = load_kimore_exact(
        "D:/Rehabilation/KIMORE_processed/Exercise1"
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=1, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=1, shuffle=False,
    )

    model = PointCloudTransformerRegressor(
        seq_len=100, num_joints=25, num_channels=3,
        dim=256, spatial_depth=6, temporal_depth=3, heads=4,
        dropout=0.1, k=20, curve_setting="default",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    loss_fn = nn.HuberLoss(reduction="mean", delta=0.1)

    best_mad_scaled = float("inf")
    best_mad_orig = float("inf")
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, 201):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).float()
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mad_scaled = evaluate_mad_scaled(model, test_loader, device)
        mad_orig = evaluate_original_scale(model, test_loader, device, sc_y)

        if mad_scaled < best_mad_scaled:
            best_mad_scaled = mad_scaled
            best_mad_orig = mad_orig
            best_epoch = epoch
            # Save best
            torch.save(model.state_dict(), "D:/Rehabilation/outputs/exact_best.pt")

        if epoch % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}: scaled MAD={mad_scaled:.4f} orig MAD={mad_orig:.4f} "
                  f"(best scaled={best_mad_scaled:.4f} orig={best_mad_orig:.4f} @ ep {best_epoch}) "
                  f"[{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Best: scaled MAD={best_mad_scaled:.4f} original MAD={best_mad_orig:.4f} at epoch {best_epoch}")
    print(f"Paper: MAD=0.185")

if __name__ == "__main__":
    main()

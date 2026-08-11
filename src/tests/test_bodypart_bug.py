"""Test if original repo's body_part bug affects MAD.

Original repo body_parts has index_Ankle_Right (72) TWICE, missing index_Foot_Right (76).
Our code has the CORRECT indices with Foot_Right.
"""
import sys, time
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (portable; file now in src/tests/)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models_curvenet import PointCloudTransformerRegressor
from train_reproduce import reshape_to_sequences, performance_metrics, seed_everything

# Original repo's BUGGY body parts (Ankle_Right twice, no Foot_Right)
BUGGY_BODY_PARTS = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36,
    40, 44, 48, 52, 56, 60, 64, 68, 72, 72,  # BUG: 72 twice, missing 76
    80, 84, 88, 92, 96,
]

# Our CORRECT body parts
CORRECT_BODY_PARTS = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36,
    40, 44, 48, 52, 56, 60, 64, 68, 72, 76,  # CORRECT: 76 = Foot_Right
    80, 84, 88, 92, 96,
]

def extract_body_parts(raw, body_parts):
    n_rows = raw.shape[0]
    n_cols = len(body_parts) * 3
    extracted = np.zeros((n_rows, n_cols), dtype=np.float32)
    for row in range(n_rows):
        counter = 0
        for part in body_parts:
            for ch in range(3):
                extracted[row, counter + ch] = raw[row, part + ch]
            counter += 3
    return extracted

def load_and_test(body_parts, label):
    print(f"\n=== {label} ===")
    print(f"  Joint 19 index: {body_parts[19]} (72=Ankle_Right, 76=Foot_Right)")
    
    seed_everything(420)
    train_x_raw = pd.read_csv("D:/Rehabilation/KIMORE_processed/Exercise1/Train_X.csv", header=None).values
    train_y_raw = pd.read_csv("D:/Rehabilation/KIMORE_processed/Exercise1/Train_Y.csv", header=None).values
    n_samples = train_x_raw.shape[0] // 100
    
    X = extract_body_parts(train_x_raw, body_parts)
    y = train_y_raw.reshape(-1, 1)
    
    sc1 = StandardScaler()
    X_scaled = sc1.fit_transform(X)
    sc2 = StandardScaler()
    y_scaled = sc2.fit_transform(y)
    
    X_4d = reshape_to_sequences(X_scaled, n_samples, 100, 25, 3)
    X_train, X_test, y_train, y_test = train_test_split(X_4d, y_scaled, test_size=0.2, random_state=420)
    
    print(f"  X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=1, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=1, shuffle=False,
    )
    
    device = torch.device("cuda")
    model = PointCloudTransformerRegressor(
        seq_len=100, num_joints=25, num_channels=3,
        dim=256, spatial_depth=6, temporal_depth=3, heads=4,
        dropout=0.1, k=20, curve_setting="default",
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    loss_fn = nn.HuberLoss(reduction="mean", delta=0.1)
    
    best_mad = float("inf")
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
        
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device).float(), y.to(device).float()
                pred = model(x)
                preds.extend(pred.cpu().numpy().flatten())
                trues.extend(y.cpu().numpy().flatten())
        mad, rmse, _, mape = performance_metrics(trues, preds)
        
        if mad < best_mad:
            best_mad = mad
            best_epoch = epoch
        
        if epoch % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}: MAD={mad:.4f} (best={best_mad:.4f} @ ep {best_epoch}) [{elapsed:.0f}s]")
    
    print(f"  Result: best MAD={best_mad:.4f} at epoch {best_epoch}")
    return best_mad

if __name__ == "__main__":
    print("Testing if original repo's body_part bug affects MAD")
    print("Original: Ankle_Right (72) twice, no Foot_Right (76)")
    print("Ours: Correct indices with Foot_Right (76)")
    
    device = torch.device("cuda")
    print(f"Device: {device}")
    
    buggy_mad = load_and_test(BUGGY_BODY_PARTS, "BUGGY (original repo)")
    correct_mad = load_and_test(CORRECT_BODY_PARTS, "CORRECT (our code)")
    
    print(f"\n=== COMPARISON ===")
    print(f"  Buggy MAD:  {buggy_mad:.4f}")
    print(f"  Correct MAD: {correct_mad:.4f}")
    print(f"  Paper MAD:   0.185")

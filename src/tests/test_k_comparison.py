"""Compare k=9 vs k=20 for 200 epochs on Exercise 1."""
import sys, time
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (portable; file now in src/tests/)
from train_reproduce import load_kimore_exercise, seed_everything, performance_metrics
import torch
import torch.nn as nn
from models_curvenet import PointCloudTransformerRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = {}

for k_val in [9, 20]:
    print(f"\n=== Testing k={k_val} ===")
    seed_everything(420)
    train_x, test_x, train_y, test_y, sc_x, sc_y = load_kimore_exercise(
        "D:/Rehabilation/KIMORE_processed/Exercise1"
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=1, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
        batch_size=1, shuffle=False,
    )

    model = PointCloudTransformerRegressor(
        seq_len=100, num_joints=25, num_channels=3,
        dim=256, spatial_depth=6, temporal_depth=3, heads=4,
        dropout=0.1, k=k_val, curve_setting="default",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

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

    elapsed = time.time() - t0
    results[k_val] = {"best_mad": best_mad, "best_epoch": best_epoch, "final_mad": mad, "time": elapsed}
    print(f"  k={k_val}: best MAD={best_mad:.4f} at epoch {best_epoch} ({elapsed:.0f}s)")

print("\n=== SUMMARY ===")
for k_val, r in results.items():
    print(f"  k={k_val}: best MAD={r['best_mad']:.4f} at epoch {r['best_epoch']}")

"""Quick k comparison: 10 epochs each."""
import sys, time
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (portable; file now in src/tests/)
from train_reproduce import load_kimore_exercise, seed_everything, performance_metrics
import torch
import torch.nn as nn
from models_curvenet import PointCloudTransformerRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

for k_val in [9, 20]:
    print(f"\n=== k={k_val} ===")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    loss_fn = nn.HuberLoss(reduction="mean", delta=0.1)
    t0 = time.time()
    for epoch in range(1, 11):
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
        print(f"  Epoch {epoch}: MAD={mad:.4f} RMSE={rmse:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

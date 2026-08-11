"""Test: train with ALL-data Y scaler (original) vs train-only Y scaler."""
import sys, time
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (portable; file now in src/tests/)
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models_curvenet import PointCloudTransformerRegressor
from train_reproduce import extract_body_parts, reshape_to_sequences, performance_metrics, seed_everything

device = torch.device("cuda")
seed_everything(420)

rx = pd.read_csv("D:/Rehabilation/KIMORE_processed/Exercise1/Train_X.csv",header=None).values
ry = pd.read_csv("D:/Rehabilation/KIMORE_processed/Exercise1/Train_Y.csv",header=None).values
n=77; X=extract_body_parts(rx); y=ry.reshape(-1,1)

# X scaler: fit on ALL (same for both)
sc1=StandardScaler(); Xs=sc1.fit_transform(X)
X4=reshape_to_sequences(Xs,n,100,25,3)

# Split first
Xt,Xv,yt_raw,yv_raw=train_test_split(X4,y,test_size=0.2,random_state=420)

for mode in ["ALL_DATA", "TRAIN_ONLY"]:
    print(f"\n=== Y scaler: {mode} ===")
    seed_everything(420)
    
    if mode == "ALL_DATA":
        # Original: fit on ALL Y data (leakage)
        sc2=StandardScaler()
        yt=sc2.fit_transform(yt_raw)  # fit on all, but only transform train
        yv=sc2.transform(yv_raw)
    else:
        # Ours: fit on TRAIN only
        sc2=StandardScaler()
        yt=sc2.fit_transform(yt_raw)
        yv=sc2.transform(yv_raw)
    
    print(f"  Scaler: mean={sc2.mean_[0]:.4f} std={sc2.scale_[0]:.4f}")
    print(f"  Y train: [{yt.min():.4f}, {yt.max():.4f}]")
    print(f"  Y test:  [{yv.min():.4f}, {yv.max():.4f}]")
    
    tl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xt),torch.from_numpy(yt)),batch_size=1,shuffle=True)
    vl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xv),torch.from_numpy(yv)),batch_size=1,shuffle=False)
    
    model=PointCloudTransformerRegressor(seq_len=100,num_joints=25,num_channels=3,dim=256,spatial_depth=6,temporal_depth=3,heads=4,dropout=0.1,k=20,curve_setting="default").to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.0)
    loss_fn=nn.HuberLoss(reduction="mean",delta=0.1)
    best=999; bep=0; t0=time.time()
    
    for ep in range(1,201):
        model.train()
        for x,y in tl:
            x,y=x.to(device).float(),y.to(device).float()
            loss=loss_fn(model(x),y); opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); ps,ts=[],[]
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(device).float(),y.to(device).float()
                ps.extend(model(x).cpu().numpy().flatten()); ts.extend(y.cpu().numpy().flatten())
        mad=performance_metrics(ts,ps)[0]
        if mad<best: best=mad; bep=ep
        if ep%50==0: print(f"  ep{ep}: MAD={mad:.4f} best={best:.4f}@{bep} [{time.time()-t0:.0f}s]")
    
    # Also compute original-scale MAD
    ps_a=np.array(ps).reshape(-1,1); ts_a=np.array(ts).reshape(-1,1)
    ps_inv=sc2.inverse_transform(ps_a); ts_inv=sc2.inverse_transform(ts_a)
    mad_orig=performance_metrics(ts_inv.flatten(),ps_inv.flatten())[0]
    print(f"  DONE: scaled best={best:.4f} orig={mad_orig:.4f}")

print("\nPaper: 0.185")

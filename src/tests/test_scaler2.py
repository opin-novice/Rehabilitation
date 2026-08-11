import sys, time, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (portable; file now in src/tests/)
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models_curvenet import PointCloudTransformerRegressor
from train_reproduce import extract_body_parts, reshape_to_sequences, performance_metrics, seed_everything

device = torch.device("cuda")
print(f"Device: {device}", flush=True)
seed_everything(420)

rx = pd.read_csv("D:/Rehabilation/KIMORE_processed/Exercise1/Train_X.csv",header=None).values
ry = pd.read_csv("D:/Rehabilation/KIMORE_processed/Exercise1/Train_Y.csv",header=None).values
X=extract_body_parts(rx); y=ry.reshape(-1,1)
sc1=StandardScaler(); Xs=sc1.fit_transform(X)
X4=reshape_to_sequences(Xs,77,100,25,3)
Xt,Xv,yt_raw,yv_raw=train_test_split(X4,y,test_size=0.2,random_state=420)

for mode in ["ALL_DATA", "TRAIN_ONLY"]:
    print(f"\n=== {mode} ===", flush=True)
    seed_everything(420)
    sc2=StandardScaler()
    if mode == "ALL_DATA":
        all_y = np.concatenate([yt_raw, yv_raw], axis=0)
        sc2.fit(all_y)
        yt=sc2.transform(yt_raw); yv=sc2.transform(yv_raw)
    else:
        yt=sc2.fit_transform(yt_raw); yv=sc2.transform(yv_raw)
    print(f"  mean={sc2.mean_[0]:.2f} std={sc2.scale_[0]:.2f}", flush=True)
    
    tl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xt),torch.from_numpy(yt)),batch_size=1,shuffle=True)
    vl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xv),torch.from_numpy(yv)),batch_size=1,shuffle=False)
    
    model=PointCloudTransformerRegressor(seq_len=100,num_joints=25,num_channels=3,dim=256,spatial_depth=6,temporal_depth=3,heads=4,dropout=0.1,k=20,curve_setting="default").to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.0)
    loss_fn=nn.HuberLoss(reduction="mean",delta=0.1)
    best=999; t0=time.time()
    for ep in range(1,101):
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
        if mad<best: best=mad
        if ep%25==0: print(f"  ep{ep}: MAD={mad:.4f} best={best:.4f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"  DONE: best={best:.4f}", flush=True)
print("\nPaper: 0.185", flush=True)

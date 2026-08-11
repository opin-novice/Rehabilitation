import sys, time
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (portable; file now in src/tests/)
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models_curvenet import PointCloudTransformerRegressor
from train_reproduce import reshape_to_sequences, performance_metrics, seed_everything

BUGGY = [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,72,80,84,88,92,96]
CORRECT = [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96]

def extract(raw, bp):
    n = raw.shape[0]; c = len(bp)*3
    out = np.zeros((n,c), dtype=np.float32)
    for r in range(n):
        cnt=0
        for p in bp:
            for ch in range(3): out[r,cnt+ch]=raw[r,p+ch]
            cnt+=3
    return out

device = torch.device("cuda")
rx = pd.read_csv("D:/Rehabilation/KIMORE_processed/Exercise1/Train_X.csv",header=None).values
ry = pd.read_csv("D:/Rehabilation/KIMORE_processed/Exercise1/Train_Y.csv",header=None).values

for label, bp in [("BUGGY", BUGGY), ("CORRECT", CORRECT)]:
    print(f"=== {label} (joint19 idx={bp[19]}) ===")
    seed_everything(420)
    X = extract(rx, bp); y = ry.reshape(-1,1)
    sc1=StandardScaler(); Xs=sc1.fit_transform(X)
    sc2=StandardScaler(); ys=sc2.fit_transform(y)
    X4=reshape_to_sequences(Xs,77,100,25,3)
    Xt,Xv,yt,yv=train_test_split(X4,ys,test_size=0.2,random_state=420)
    tl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xt),torch.from_numpy(yt)),batch_size=1,shuffle=True)
    vl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xv),torch.from_numpy(yv)),batch_size=1,shuffle=False)
    
    model=PointCloudTransformerRegressor(seq_len=100,num_joints=25,num_channels=3,dim=256,spatial_depth=6,temporal_depth=3,heads=4,dropout=0.1,k=20,curve_setting="default").to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.0)
    loss_fn=nn.HuberLoss(reduction="mean",delta=0.1)
    best=999; t0=time.time()
    for ep in range(1,21):
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
        if ep%5==0: print(f"  ep{ep}: MAD={mad:.4f} best={best:.4f} [{time.time()-t0:.0f}s]")
    print(f"  DONE: best={best:.4f}")
print("Paper: 0.185")

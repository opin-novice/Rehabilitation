"""Run remaining reviewer analyses: sensor-ID, few-shot, partial FT."""
import sys, os, json, glob
sys.path.insert(0, "src")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models_stgcn import TCNRegressor
from selfsup.data import load_corpus_with_labels

RESULTS_DIR = "archive/legacy_results/kimore_loso_78fold"
PRETRAIN_DIR = "outputs/ssl_pretrain"
OUT_PATH = "results/reviewer_analyses.json"

def rebuild(cp):
    c = torch.load(cp, map_location="cpu")
    a = c.get("args", {})
    m = TCNRegressor(seq_len=100, d_model=a.get("d_model",128),
                     num_blocks=a.get("tcn_blocks",4), dropout=a.get("dropout",0.3))
    m.load_state_dict(c["model_state"], strict=False)
    m.eval()
    return m

def extract(model, X, batch=256):
    xt = torch.from_numpy(X.astype(np.float32))
    feats = []
    for i in range(0, len(xt), batch):
        feats.append(model.forward_features(xt[i:i+batch]).detach().cpu().numpy())
    return np.concatenate(feats) if feats else np.array([])

def train_kimore(model, X, y, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=16, shuffle=True)
    best, stale = float("inf"), 0
    for ep in range(1, epochs+1):
        model.train(); lss=0
        for xb, yb in dl:
            opt.zero_grad()
            p = model(xb)
            if isinstance(p, tuple): p=p[0]
            l = torch.nn.functional.mse_loss(p, yb)
            l.backward(); opt.step(); lss+=l.item()*len(xb)
        lss/=len(xt)
        if lss < best: best=lss; stale=0
        else: stale+=1
        if stale>=50: break
    model.eval()

def main():
    X_rehab, labels_rehab, _ = load_corpus_with_labels("REHAB246")
    X_uiprmd, labels_uiprmd, _ = load_corpus_with_labels("UIPRMD")
    X_kimore, y_kimore, _ = load_corpus_with_labels("KIMORE")

    if os.path.isfile(OUT_PATH):
        with open(OUT_PATH) as f: all_results = json.load(f)
    else:
        all_results = {}

    ckpts = sorted(glob.glob(f"{RESULTS_DIR}/A_scratch/fold_*/best_model.pt"))[:5]
    datasets = [("KIMORE",X_kimore),("REHAB246",X_rehab),("UIPRMD",X_uiprmd)]
    lm = {"KIMORE":0,"REHAB246":1,"UIPRMD":2}
    a3, akr, aku = [], [], []
    for cp in ckpts:
        m = rebuild(cp)
        src = {}
        Fs, ys = [], []
        for n,d in datasets:
            feat = extract(m,d); src[n]=feat
            Fs.append(feat); ys.append(np.full(len(feat),lm[n]))
        F=np.concatenate(Fs); y=np.concatenate(ys)
        a3.append(float(np.mean(cross_val_score(LogisticRegression(max_iter=1000,C=1),F,y,cv=3,scoring="balanced_accuracy"))))
        for a,b,k in [("KIMORE","REHAB246",akr),("KIMORE","UIPRMD",aku)]:
            F2=np.concatenate([src[a],src[b]]); y2=np.array([0]*len(src[a])+[1]*len(src[b]))
            k.append(float(np.mean(cross_val_score(LogisticRegression(max_iter=1000,C=1),F2,y2,cv=3,scoring="balanced_accuracy"))))
    all_results["sensor_id_probe"] = {
        "mean_3way_balanced_acc": float(np.mean(a3)),
        "std_3way_balanced_acc": float(np.std(a3)),
        "mean_kimore_vs_rehab246_balanced_acc": float(np.mean(akr)),
        "mean_kimore_vs_uiprmd_balanced_acc": float(np.mean(aku)),
        "chance_3way": 1/3, "chance_2way": 0.5,
    }
    print(f"Sensor-ID: 3way={np.mean(a3):.3f} KR={np.mean(akr):.3f} KU={np.mean(aku):.3f}")

    for cond_dir in ["A_scratch","B_contrastive_lp","C_contrastive_ft","D_masked_lp","E_masked_ft"]:
        cpts = sorted(glob.glob(f"{RESULTS_DIR}/{cond_dir}/fold_*/best_model.pt"))[:5]
        if not cpts: continue
        Fr, Fu, Fsrc = [], [], None
        for cp in cpts:
            m = rebuild(cp)
            if Fsrc is None: Fsrc=extract(m,X_kimore)
            Fr.append(extract(m,X_rehab))
            Fu.append(extract(m,X_uiprmd))
        Fr=np.mean(Fr,axis=0); Fu=np.mean(Fu,axis=0)
        sc = StandardScaler()
        Fsrc_s = sc.fit_transform(Fsrc)
        for cn,Ft,la in [("REHAB246",Fr,labels_rehab),("UIPRMD",Fu,labels_uiprmd)]:
            Fts = sc.transform(Ft)
            for n in [1,5,10,20]:
                na=[]
                for sd in range(3):
                    rng=np.random.RandomState(sd)
                    idx=rng.choice(len(Fts),n,replace=False)
                    msk=np.zeros(len(Fts),dtype=bool);msk[idx]=True
                    if len(np.unique(la[msk]))<2: continue
                    clf=LogisticRegression(max_iter=1000,C=1.0)
                    clf.fit(Fts[msk],la[msk])
                    p=clf.predict_proba(Fts[~msk])[:,1]
                    t=la[~msk]
                    if len(np.unique(t))>1:
                        try: na.append(float(max(roc_auc_score(t,p),1.0-roc_auc_score(t,p))))
                        except: pass
                k=f"fewshot/{cond_dir}/{cn}"
                if k not in all_results: all_results[k]={}
                all_results[k][f"n{n}"]={"mean_auroc":float(np.mean(na)) if na else None}
        print(f"  fewshot {cond_dir} done")

    for ssl in ["contrastive","masked"]:
        cpth = f"{PRETRAIN_DIR}/all_corpora/{ssl}_encoder.pt"
        if not os.path.isfile(cpth): continue
        es = torch.load(cpth, map_location="cpu")
        es = es.get("encoder_state", es.get("model_state", {}))
        for cn,Xc,la in [("REHAB246",X_rehab,labels_rehab),("UIPRMD",X_uiprmd,labels_uiprmd)]:
            for fm,desc in [("freeze_proj_b01","proj+b0-1"),("freeze_proj","proj")]:
                m=TCNRegressor(seq_len=100,d_model=128,num_blocks=4,dropout=0.3)
                m.load_state_dict(es,strict=False)
                for n,p in m.named_parameters():
                    if fm=="freeze_proj_b01" and (n.startswith("input_proj") or n.startswith("blocks.0") or n.startswith("blocks.1")):
                        p.requires_grad=False
                    elif fm=="freeze_proj" and n.startswith("input_proj"):
                        p.requires_grad=False
                train_kimore(m,X_kimore,y_kimore)
                xt=torch.from_numpy(Xc.astype(np.float32))
                with torch.no_grad():
                    preds=m(xt).squeeze(-1).numpy()
                sd=float(np.std(preds))
                auroc=None
                if la is not None and len(np.unique(la))>1:
                    try:
                        aa=roc_auc_score(la,preds)
                        auroc=float(max(aa,1.0-aa))
                    except: pass
                k=f"partial_ft/{ssl}_{fm}/{cn}"
                all_results[k]={"mean_auroc":auroc,"pred_sd":sd}
                print(f"  {k}: AUROC={auroc}")

    Path(OUT_PATH).parent.mkdir(parents=True,exist_ok=True)
    Path(OUT_PATH).write_text(json.dumps(all_results,indent=2))
    print(f"Saved to {OUT_PATH}")

if __name__=="__main__":
    main()
## **Full Replication and Implementation Guide** 

For King-Rafat/Transformer_Rehabilitation: cloning, environment setup, data preparation, training, evaluation, and visualization 

**Deliverables included:** this PDF plus a companion code bundle ZIP containing runnable starter scripts. 

**Important honesty note:** I inspected the public repository and found that it is not plug-and-play. It has no complete run instructions, no pinned dependency file, and the README dataset links are partially inconsistent with the rehabilitation task. This guide gives you a working replication path that keeps the repository idea but fills the missing engineering pieces. 

Figure 1. Practical replication pipeline. 

Transformer_Rehabilitation replication guide 

Page 1 

## **1. What the repository contains** 

The public GitHub repository is titled **Transformer Network for the Assessment of Physical Rehabilitation Exercise** . The README says the approach uses curve-based data aggregation for feature augmentation and then fuses it with a Transformer architecture for rehabilitation exercise quality assessment. The visible repository contains: **Data_Proc/data_processing.py** , an **Images** folder, **core** model code, **README.md** , and **Rehabilitation.ipynb** . 

|**Item**|**Role in implementation**|**Risk / action**|
|---|---|---|
|README.md|Project description and minimal dependency notes|How-to-run section is empty; follow this guide inst|
|Rehabilitation.ipynb|Notebook imports PyTorch, CurveNet, Captum, sklearn, an|d plotting tools<br>Use after environment is ready; convert cells to sc|
|Data_Proc/data_process|ing.py<br>Loads KIMORE-like Train_X.csv and Train_Y.csv; reshape|s to [N, 100, 25, 3]<br>Very format-dependent; use the cleaned loader in|
|core/models|CurveNet / Curve Aggregation implementation adapted fro|m point-cloud work<br>Patch npoint values for 25-joint skeleton data befo|
|core/main_cls.py|ModelNet40 point-cloud classification training loop|Useful reference, but not directly the rehabilitation|



The original README says dependencies are python>=3.7, PyTorch, and cudatoolkit>11.2. For a new machine in 2026, use a modern Python and PyTorch install from the official PyTorch selector rather than trying to reproduce very old CUDA pins. 

## **2. Clone the repository** 

Use one of the following command sets. For beginners, create a clean folder with no spaces in the path, such as D:\research or ~/research. 

## **Linux / WSL / macOS** 

```
mkdir -p ~/research
cd ~/research
git clone https://github.com/King-Rafat/Transformer_Rehabilitation.git
cd Transformer_Rehabilitation
git status
ls
```

## **Windows PowerShell** 

```
mkdir D:\research
cd D:\research
git clone https://github.com/King-Rafat/Transformer_Rehabilitation.git
cd Transformer_Rehabilitation
git status
dir
```

Expected result: you should see Data_Proc, Images, core, README.md, and Rehabilitation.ipynb. 

## **3. Create the environment** 

The current official PyTorch installation page shows that the stable pip command should be selected according to your OS and CUDA version. At the time checked, the selector showed a CUDA 12.6 pip command, but you should always verify the PyTorch page before installing on a new GPU. 

## **Recommended Python environment** 

```
# Linux / WSL
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

```
# Then install PyTorch from the official selector. Example CUDA 12.6 command:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

```
# Install research dependencies
```

```
pip install numpy pandas scikit-learn scipy matplotlib tqdm h5py einops captum pyyaml
```

## **Windows PowerShell** 

```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy pandas scikit-learn scipy matplotlib tqdm h5py einops captum pyyaml
```

## **Verify GPU access** 

```
python - <<'PYCHECK'
import torch
```

Transformer_Rehabilitation replication guide 

Page 2 

```
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PYCHECK
```

Transformer_Rehabilitation replication guide 

Page 3 

## **4. Repository-specific fixes before training** 

These fixes turn the repository from a research notebook into a reproducible project. 

G[Add a requirements.txt file so every teammate installs the same packages.] 

G[Do not train directly from the notebook first. Start with a script and a tiny dummy dataset to test shape flow.] 

G[Keep raw datasets outside Git and only store small processed CSV samples for testing.] 

- G[Patch CurveNet only if you use the original CurveNet frame encoder. Its visible default npoint=117 is not appropriate] for a 25-joint skeleton input. 

G[Use regression metrics for rehabilitation scoring: RMSE, MAE, R2, Pearson/Spearman correlation when relevant.] 

## **Create folders** 

```
mkdir -p KIMORE data checkpoints outputs logs notebooks
mkdir -p outputs/smoke_test outputs/real_run
```

## **Optional CurveNet skeleton patch** 

```
# Run only if you choose --encoder curvenet
python ../rehab_replication_code/src/patch_curvenet_for_skeleton.py --file
    core/models/curvenet_cls.py
```

Safer beginner choice: first use the included PointFrameEncoder + Transformer model. After that works, switch to the repository CurveNet encoder and compare results. 

## **5. Dataset choices and preparation** 

The repo mentions UI-PRMD, KIMORE, and IRDS. For replication, start with UI-PRMD or KIMORE because they are common in physical rehabilitation movement assessment literature. The UI-PRMD official site describes 10 rehabilitation movements performed by 10 healthy subjects, repeated 10 times, captured with Vicon and Kinect, with positions and angles. It also provides segmented and reduced data variants. KIMORE is clinically stronger for scoring because it includes low-back-pain rehabilitation exercises, RGB/depth/skeleton inputs, physician-defined features, and clinical scores. 

|**Dataset**|**Best use**|**What to download / prepare**|
|---|---|---|
|UI-PRMD|Beginner-friendly movement correctness / reduced sc|oring experiments<br>Use segmented movements or reduced dataset. Convert|
|KIMORE|Clinical quality-score regression and remote rehab as|sessment<br>Use skeleton joint positions plus clinical score/features.|
|IRDS / IntelliRehabDS|Correct vs incorrect classification; external validation|Use after baseline works to test cross-dataset generaliza|



## **Target processed format** 

```
Transformer_Rehabilitation/
  KIMORE/
    Exercise1/
      Train_X.csv   # one timestep per row
      Train_Y.csv   # one score per sample
    Exercise2/
      Train_X.csv
      Train_Y.csv
```

Transformer_Rehabilitation replication guide 

Page 4 

Figure 2. Data tensor flow expected by the implementation. 

Transformer_Rehabilitation replication guide 

Page 5 

## **6. Smoke test before real training** 

Before spending time on real datasets, confirm that the environment, model, loader, outputs, and plots work on synthetic skeleton data. 

`# From the cloned Transformer_Rehabilitation folder python ../rehab_replication_code/src/make_dummy_data.py --out_dir dummy_data/Exercise1 --samples 40` 

- `python ../rehab_replication_code/src/train.py   --data_dir dummy_data/Exercise1   --epochs 3 --batch_size 8   --out_dir outputs/smoke_test` 

## **Expected files after smoke test** 

`outputs/smoke_test/ best_model.pt training_curves.png prediction_scatter.png history.npy` 

Figure 3. Example training curve output. 

## **7. Real training run** 

After you prepare real Train_X.csv and Train_Y.csv files, train with conservative settings first. Increase batch size only after confirming VRAM stability. 

`python ../rehab_replication_code/src/train.py   --data_dir KIMORE/Exercise1   --epochs 50 --batch_size 8   --lr 0.0001   --encoder pointnet   --out_dir outputs/kimore_ex1_point_transformer` 

## **Optional CurveNet encoder run** 

`python ../rehab_replication_code/src/patch_curvenet_for_skeleton.py --file core/models/curvenet_cls.py python ../rehab_replication_code/src/train.py   --data_dir KIMORE/Exercise1   --epochs 50 --batch_size 4   --lr 0.00005   --encoder curvenet   --repo_root .   --out_dir outputs/kimore_ex1_curvenet_transformer` 

## **Evaluation run** 

`python ../rehab_replication_code/src/evaluate.py   --checkpoint` 

`outputs/kimore_ex1_point_transformer/best_model.pt   --data_dir KIMORE/Exercise1   --out_dir outputs/kimore_ex1_eval` 

Figure 4. Example metric summary. Replace with real run values. 

Transformer_Rehabilitation replication guide 

Page 6 

## **8. Visualization outputs you should generate** 

For research reporting, do not only show numbers. Save visual evidence for model behavior and data sanity. 

G[Skeleton frame plot for random samples to confirm the joint order is correct.] 

G[Training and validation loss curve to diagnose overfitting.] 

G[Ground-truth versus prediction scatter plot for score regression.] 

G[Residual plot by exercise type or subject to detect bias.] 

G[Ablation bar chart comparing point-only Transformer, CurveNet+Transformer, and graph baselines.] 

Figure 5. Skeleton visualization template. 

## **Skeleton visualization command** 

- `# Use visualize.plot_skeleton_frame(frame_xyz, out_path)` 

- `# frame_xyz must be shaped [25, 3]. # Example inside a notebook: from rehab_replication_code.src.visualize import plot_skeleton_frame plot_skeleton_frame(x[0, 0], 'outputs/skeleton_frame.png')` 

## **9. Hardware and resource settings** 

For 12 GB VRAM, start with batch_size=4 or 8, transformer_dim=256, layers=2, nhead=4, and mixed precision only after the normal run works. For 8 GB VRAM, use batch_size=2 or 4. For CPU-only training, run a tiny smoke test only; full experiments will be slow. 

Transformer_Rehabilitation replication guide 

Page 7 

|**Issue**|**Symptom**|**Fix**|
|---|---|---|
|CUDA out of memory|Training crashes during forward/backward|Reduce batch size, transformer_dim, layers, or use --amp.|
|Shape mismatch|Expected [B,T,J,C] but got another shape|Check seq_len and Train_X rows; rows must equal samples x|
|Bad skeleton plot|Human body looks disconnected|Joint order or coordinate columns are wrong. Update JOINT_S|
|No learning|Flat loss curve|Normalize scores, check labels, reduce LR, verify train/validati|
|CurveNet failure|npoint or grouping error|Patch npoint to <= number of joints, lower k, or use PointFram|



Transformer_Rehabilitation replication guide 

Page 8 

## **10. Complete implementation sequence checklist** 

G[Install Git, Python 3.10/3.11, NVIDIA driver, and CUDA-compatible PyTorch.] 

G[Clone the GitHub repository.] 

- G[Create and activate a virtual environment.] 

G[Install the companion requirements and verify torch.cuda.is_available().] 

G[Create KIMORE/data/checkpoints/outputs folders.] 

G[Run dummy-data smoke test for 3 epochs.] 

G[Download UI-PRMD or KIMORE from the official sources.] 

G[Convert raw skeleton files into Train_X.csv and Train_Y.csv with fixed seq_len=100.] 

G[Plot 5 random skeleton frames to confirm joint order.] 

- G[Train PointFrameEncoder + Transformer baseline.] 

G[Evaluate RMSE, MAE, R2, and prediction scatter.] 

- G[Patch and train CurveNet + Transformer optional variant.] 

G[Run ablations: no Transformer, no frame encoder, different sequence lengths, different splits.] 

G[Save all configs, random seeds, model checkpoints, plots, and logs.] 

G[Write results table and compare with published baselines.] 

## **11. Research-quality experiment table** 

|**Experiment**|**Encoder**|**Temporal model**|**Metric target**|**Why run it**|are learnable.<br>rison.<br>seen people.|
|---|---|---|---|---|---|
|E1 baseline|PointFrameEncoder|Mean pooling|RMSE/MAE|Shows whether data and labels||
|E2 main|PointFrameEncoder|Transformer|RMSE/MAE/R2|Clean, stable reproduction.||
|E3 repo-inspired|CurveNet patched|Transformer|RMSE/MAE/R2|Closest to repository idea.||
|E4 graph baseline|ST-GCN or simple GCN|GRU/Transformer|RMSE/MAE/R2|Strong skeleton-specific compa||
|E5 robustness|Best model|Best temporal model|Cross-subject metrics|Tests if model generalizes to un||



Transformer_Rehabilitation replication guide 

Page 9 

## **12. Code appendix: commands and key files** 

The companion ZIP contains the full code. The most important files are reproduced below so the PDF is self-contained for implementation. 

## **requirements.txt** 

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
tqdm>=4.66
captum>=0.7
pyyaml>=6.0
h5py>=3.9
einops>=0.7
# Install torch/torchvision from https://pytorch.org/get-started/locally/ for your CUDA version.
# Example currently shown by PyTorch for CUDA 12.6:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## **src/rehab_dataset.py** 

```
import os
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
# Joint starts used in the original repository's Data_Proc/data_processing.py.
# Each joint occupies x,y,z plus one orientation/extra slot in many Kinect exports,
# therefore the code jumps by 4 and then keeps the first 3 coordinate columns.
JOINT_STARTS = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
    48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96,
]
@dataclass
class DatasetStats:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
class SkeletonCSVRegressionDataset(Dataset):
    """Load preprocessed KIMORE/UI-PRMD style CSV files.
    Expected:
    - Train_X.csv: one timestep per row, with enough columns to contain 25 joints.
    - Train_Y.csv: one scalar clinical/quality score per sample.
    - seq_len: default 100 timesteps per sample, matching the original repo loader.
    Output:
    x: Tensor [T, J, C]
    y: Tensor [1]
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int = 100,
        fit_scaler: bool = True,
        x_scaler: Optional[StandardScaler] = None,
        y_scaler: Optional[StandardScaler] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = len(JOINT_STARTS)
        self.num_channels = 3
        x = self._select_xyz_columns(x).astype(np.float32)
        y = y.reshape(-1, 1).astype(np.float32)
        if x.shape[0] % seq_len != 0:
            raise ValueError(
                f"Train_X rows ({x.shape[0]}) must be divisible by seq_len ({seq_len}). "
                "Check segmentation or change --seq_len."
            )
        n_samples = x.shape[0] // seq_len
        if y.shape[0] != n_samples:
            raise ValueError(
                f"Train_Y has {y.shape[0]} rows but Train_X implies {n_samples} samples."
            )
```

```
        self.x_scaler = x_scaler or StandardScaler()
```

Transformer_Rehabilitation replication guide 

Page 10 

```
        self.y_scaler = y_scaler or StandardScaler()
        if fit_scaler:
            x = self.x_scaler.fit_transform(x)
            y = self.y_scaler.fit_transform(y)
        else:
            x = self.x_scaler.transform(x)
            y = self.y_scaler.transform(y)
        self.x = x.reshape(n_samples, seq_len, self.num_joints, self.num_channels)
        self.y = y
    @staticmethod
    def _select_xyz_columns(raw: np.ndarray) -> np.ndarray:
        cols = []
        for start in JOINT_STARTS:
            cols.extend([start, start + 1, start + 2])
        if raw.shape[1] <= max(cols):
            raise ValueError(
                f"Train_X has {raw.shape[1]} columns, but this loader needs column {max(cols)}. "
                "If your raw data already has 25*3 columns, adjust JOINT_STARTS or use the fallback
                    below."
            )
        return raw[:, cols]
    def __len__(self) -> int:
        return self.x.shape[0]
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])
def load_csv_arrays(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    x_path = os.path.join(data_dir, "Train_X.csv")
    y_path = os.path.join(data_dir, "Train_Y.csv")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing {y_path}")
    x = pd.read_csv(x_path, header=None).values
    y = pd.read_csv(y_path, header=None).values.squeeze()
    return x, y
def make_dataloaders(
    data_dir: str,
    seq_len: int = 100,
    batch_size: int = 8,
    test_size: float = 0.2,
    seed: int = 145,
    num_workers: int = 0,
):
    raw_x, raw_y = load_csv_arrays(data_dir)
    sample_ids = np.arange(raw_y.reshape(-1).shape[0])
    train_ids, val_ids = train_test_split(sample_ids, test_size=test_size, random_state=seed)
    def slice_rows(ids):
        rows = []
        for i in ids:
            rows.extend(range(i * seq_len, (i + 1) * seq_len))
        return raw_x[rows], raw_y[ids]
    train_x, train_y = slice_rows(train_ids)
    val_x, val_y = slice_rows(val_ids)
    train_ds = SkeletonCSVRegressionDataset(train_x, train_y, seq_len=seq_len, fit_scaler=True)
    val_ds = SkeletonCSVRegressionDataset(
        val_x,
        val_y,
        seq_len=seq_len,
        fit_scaler=False,
        x_scaler=train_ds.x_scaler,
        y_scaler=train_ds.y_scaler,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds
```

## **src/models.py** 

```
import math
from typing import Optional
```

```
import torch
import torch.nn as nn
```

```
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
```

Transformer_Rehabilitation replication guide 

Page 11 

```
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
```

```
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]
```

```
class PointFrameEncoder(nn.Module):
    """Simple point/joint encoder for each frame.
```

```
    This is the safest beginner baseline. It is not as fancy as CurveNet, but it
    uses the same idea of learning local joint features before temporal modeling.
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64), nn.ReLU(),
            nn.Linear(64, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, J, C]
        x = self.net(x)                 # [B, T, J, D]
        x = x.max(dim=2).values         # [B, T, D]
        return x
```

```
class RepoCurveNetFrameEncoder(nn.Module):
    """Optional wrapper around the original repo's CurveNet encoder.
```

```
    Important: the original core/models/curvenet_cls.py was adapted from ModelNet40
    point-cloud classification and uses npoint=117 in several CIC blocks. For a 25-joint
    skeleton, patch those npoint values to <=25 before using this wrapper.
    """
    def __init__(self, repo_root: str, out_dim: int = 256):
        super().__init__()
        import sys
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from core.models.curvenet_cls import CurveNet
        self.backbone = CurveNet()
        self.proj = nn.Linear(256, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, J, C] -> CurveNet expects [B*T, C, J]
        b, t, j, c = x.shape
        y = x.reshape(b * t, j, c).permute(0, 2, 1).contiguous()
        feat = self.backbone(y)          # expected [B*T, 256, P]
        feat = feat.mean(dim=-1)         # [B*T, 256]
        feat = self.proj(feat).reshape(b, t, -1)
        return feat
class SkeletonTransformerRegressor(nn.Module):
    def __init__(
        self,
        seq_len: int = 100,
        encoder_dim: int = 256,
        transformer_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.frame_encoder = encoder or PointFrameEncoder(out_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, transformer_dim)
        self.pos = PositionalEncoding(transformer_dim, max_len=seq_len + 10)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, J, C]
        x = self.frame_encoder(x)
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.temporal(x)
        x = x.mean(dim=1)
        return self.head(x)
```

```
def build_model(args):
```

Transformer_Rehabilitation replication guide 

Page 12 

```
    if args.encoder == "curvenet":
```

```
        encoder = RepoCurveNetFrameEncoder(repo_root=args.repo_root, out_dim=args.encoder_dim)
    else:
```

```
        encoder = PointFrameEncoder(out_dim=args.encoder_dim)
    return SkeletonTransformerRegressor(
        seq_len=args.seq_len,
        encoder_dim=args.encoder_dim,
        transformer_dim=args.transformer_dim,
        nhead=args.nhead,
        num_layers=args.layers,
        dropout=args.dropout,
        encoder=encoder,
    )
```

```
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## **src/train.py** 

```
import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
```

```
from rehab_dataset import make_dataloaders
from models import build_model, count_parameters
from visualize import plot_training_curves, plot_prediction_scatter
```

```
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```

```
def inverse_y(dataset, arr):
    arr = np.asarray(arr).reshape(-1, 1)
    return dataset.y_scaler.inverse_transform(arr).reshape(-1)
def evaluate(model, loader, dataset, device):
    model.eval()
    preds, targets = [], []
    loss_fn = nn.MSELoss()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds).reshape(-1)
    targets = np.concatenate(targets).reshape(-1)
    preds_raw = inverse_y(dataset, preds)
    targets_raw = inverse_y(dataset, targets)
    rmse = mean_squared_error(targets_raw, preds_raw, squared=False)
    mae = mean_absolute_error(targets_raw, preds_raw)
    r2 = r2_score(targets_raw, preds_raw) if len(targets_raw) > 1 else float("nan")
    return total_loss / max(n, 1), rmse, mae, r2, preds_raw, targets_raw
```

```
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder with Train_X.csv and Train_Y.csv")
    parser.add_argument("--out_dir", default="outputs/exp1")
    parser.add_argument("--repo_root", default=".", help="Original repo root; needed only for
        --encoder curvenet")
    parser.add_argument("--encoder", choices=["pointnet", "curvenet"], default="pointnet")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--transformer_dim", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=145)
```

Transformer_Rehabilitation replication guide 

Page 13 

```
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    args = parser.parse_args()
    seed_everything(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_ds = make_dataloaders(
        args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        test_size=args.test_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model = build_model(args).to(device)
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_mae": [], "val_r2": []}
    best_rmse = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = x.to(device).float()
            y = y.to(device).float()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        train_loss = total / max(n, 1)
        val_loss, rmse, mae, r2, preds, targets = evaluate(model, val_loader, train_ds, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(rmse)
        history["val_mae"].append(mae)
        history["val_r2"].append(r2)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} RMSE={rmse:.4f}
            MAE={mae:.4f} R2={r2:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out_dir,
                "best_model.pt"))
            plot_prediction_scatter(targets, preds, os.path.join(args.out_dir,
                "prediction_scatter.png"))
        plot_training_curves(history, os.path.join(args.out_dir, "training_curves.png"))
    np.save(os.path.join(args.out_dir, "history.npy"), history)
    print(f"Done. Best RMSE: {best_rmse:.4f}. Outputs saved in {args.out_dir}")
if __name__ == "__main__":
    main()
```

## **src/visualize.py** 

```
import os
import numpy as np
import matplotlib.pyplot as plt
# Approximate Kinect-style skeleton connections for 25 joints after JOINT_STARTS selection.
SKELETON_EDGES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (6, 22), (7, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (10, 24), (11, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]
def plot_training_curves(history, out_path):
    plt.figure(figsize=(8, 5))
    x = np.arange(1, len(history["train_loss"]) + 1)
    plt.plot(x, history["train_loss"], label="train loss")
    plt.plot(x, history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
```

Transformer_Rehabilitation replication guide 

Page 14 

```
    plt.ylabel("Scaled MSE loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
def plot_prediction_scatter(y_true, y_pred, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Ground-truth score")
    plt.ylabel("Predicted score")
    plt.title("Prediction quality")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
def plot_skeleton_frame(frame_xyz, out_path, title="Skeleton frame"):
    """frame_xyz shape: [25, 3]"""
    frame_xyz = np.asarray(frame_xyz)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(frame_xyz[:, 0], frame_xyz[:, 1], frame_xyz[:, 2])
    for a, b in SKELETON_EDGES:
        ax.plot(
            [frame_xyz[a, 0], frame_xyz[b, 0]],
            [frame_xyz[a, 1], frame_xyz[b, 1]],
            [frame_xyz[a, 2], frame_xyz[b, 2]],
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
```

## **src/evaluate.py** 

```
import argparse
import os
import torch
from rehab_dataset import make_dataloaders
from models import build_model
from train import evaluate
from visualize import plot_prediction_scatter
```

```
def main():
```

```
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="outputs/eval")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--encoder", choices=["pointnet", "curvenet"], default="pointnet")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--transformer_dim", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, train_ds = make_dataloaders(args.data_dir, seq_len=args.seq_len,
        batch_size=args.batch_size)
    model = build_model(args).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    val_loss, rmse, mae, r2, preds, targets = evaluate(model, val_loader, train_ds, device)
    print(f"val_loss={val_loss:.4f} RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    plot_prediction_scatter(targets, preds, os.path.join(args.out_dir,
        "prediction_scatter_eval.png"))
```

```
if __name__ == "__main__":
    main()
```

## **src/prepare_kimore_from_raw_template.py** 

```
"""Template for turning raw KIMORE/UI-PRMD skeleton files into Train_X.csv/Train_Y.csv.
```

```
You must adapt this file to the exact raw dataset folder you download.
The target format is the one expected by rehab_dataset.py:
```

- `Train_X.csv: one row per timestep; columns contain x,y,z and any extra raw columns.` 

- `Train_Y.csv: one scalar score per segmented sample.` 

Transformer_Rehabilitation replication guide 

Page 15 

```
For UI-PRMD reduced CSVs, you may already have movement-quality scores. For KIMORE,
clinical questionnaire scores/features may need to be merged by subject/exercise/repetition.
"""
```

```
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
def load_one_sequence(path: str) -> np.ndarray:
    # Common case: ASCII/CSV with comma delimiter.
    arr = pd.read_csv(path, header=None).values.astype("float32")
    return arr
```

```
def resample_to_fixed_length(arr: np.ndarray, seq_len: int = 100) -> np.ndarray:
    old_t = np.linspace(0, 1, arr.shape[0])
    new_t = np.linspace(0, 1, seq_len)
    out = np.zeros((seq_len, arr.shape[1]), dtype="float32")
    for c in range(arr.shape[1]):
        out[:, c] = np.interp(new_t, old_t, arr[:, c])
    return out
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--score_csv", required=True, help="CSV with columns: file,score")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=100)
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.score_csv)
    all_x, all_y = [], []
    for _, row in scores.iterrows():
        file_path = os.path.join(args.raw_dir, row["file"])
        seq = load_one_sequence(file_path)
        seq = resample_to_fixed_length(seq, seq_len=args.seq_len)
        all_x.append(seq)
        all_y.append(float(row["score"]))
    x = np.concatenate(all_x, axis=0)
    y = np.asarray(all_y, dtype="float32").reshape(-1, 1)
    pd.DataFrame(x).to_csv(os.path.join(args.out_dir, "Train_X.csv"), header=False, index=False)
    pd.DataFrame(y).to_csv(os.path.join(args.out_dir, "Train_Y.csv"), header=False, index=False)
    print(f"Wrote {args.out_dir}/Train_X.csv with shape {x.shape}")
    print(f"Wrote {args.out_dir}/Train_Y.csv with shape {y.shape}")
if __name__ == "__main__":
    main()
```

## **src/patch_curvenet_for_skeleton.py** 

```
"""Patch the original CurveNet classification file for 25-joint skeleton input.
```

```
Run from inside the cloned Transformer_Rehabilitation repository:
python ../rehab_replication_code/src/patch_curvenet_for_skeleton.py --file
    core/models/curvenet_cls.py
```

```
What it does:
- Backs up the original file.
- Replaces CIC npoint=117 with npoint=25.
```

```
This is a minimum adaptation. For publishable work, tune npoint/radius/k with validation.
"""
import argparse
from pathlib import Path
```

```
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="core/models/curvenet_cls.py")
    args = parser.parse_args()
    path = Path(args.file)
    text = path.read_text()
    backup = path.with_suffix(path.suffix + ".backup")
    backup.write_text(text)
    text = text.replace("npoint=117", "npoint=25")
    path.write_text(text)
    print(f"Backed up original to {backup}")
    print(f"Patched {path}: replaced npoint=117 with npoint=25")
if __name__ == "__main__":
    main()
```

Transformer_Rehabilitation replication guide 

Page 16 

## **13. Source notes checked while preparing this guide** 

G[GitHub repository: https://github.com/King-Rafat/Transformer_Rehabilitation] 

- G[Repository README raw file:] 

https://raw.githubusercontent.com/King-Rafat/Transformer_Rehabilitation/refs/heads/main/README.md 

G[Repository Data_Proc/data_processing.py and core files visible in GitHub.] 

- G[UI-PRMD official page: https://www.idahofallshighered.org/vakanski/ui-prmd.html] 

- G[KIMORE IEEE abstract page / metadata: https://ieeexplore.ieee.org/abstract/document/8736767] 

- G[PyTorch official installation selector: https://pytorch.org/get-started/locally/] 

- G[IntelliRehabDS MDPI / Zenodo references for external validation.] 

Caution: The repository README includes links to CIFAR/TinyImageNet and an unrelated Hugging Face dataset. Those are not physical rehabilitation datasets. Use the official rehabilitation dataset sources above instead. 

Transformer_Rehabilitation replication guide 

Page 17 


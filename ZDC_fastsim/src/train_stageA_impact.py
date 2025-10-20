#!/usr/bin/env python3
"""
Stage A — MLP "low budget" per predire (x_imp, y_imp) da 7 input (E, vx, vy, vz, px, py, pz).

Esecuzione tipica:
  python3.11 train_stageA_impact.py \
    --compact-dir /path/Dati_compact \
    --out-dir checkpoints_A \
    --train-shards 1
"""

from __future__ import annotations
import os, json, time, math, argparse, random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# ---- utility comuni (dal tuo utility.py) ----
from utility import seed_everything, load_json_gz

# ---- costanti di colonna ----
FEATURE_NAMES = ["E","vx","vy","vz","px","py","pz"]  # 7 input nel parquet compatto
TARGET_NAMES  = ["x_imp","y_imp"]                    # target in [0,1] nel parquet compatto

# ======================
# CLI
# ======================
def parse_args():
    ap = argparse.ArgumentParser(description="Stage A: Impact MLP (x_imp, y_imp) from 7 inputs")
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/",
                    help="Cartella compatta con train_compact/, test_compact/, stats/")
    ap.add_argument("--out-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_A",
                    help="Cartella per checkpoint e log")
    ap.add_argument("--train-shards", type=int, default=1,
                    help="Numero di shard parquet da usare per il train (default: 1)")

    # training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)

    # dataloader
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--shuffle-buffer", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument("--no-amp", action="store_true", help="Disabilita autocast AMP")

    return ap.parse_args()

# ======================
# Dataset iterabile shardato
# ======================
class ImpactIterable(IterableDataset):
    """
    Itera su più parquet: per epoca shuffla l'ordine dei file, legge un file alla volta,
    usa un buffer per shuffle parziale, normalizza gli input con mean/std globali.
    """
    def __init__(self, parquet_paths: List[Path],
                 mean7: np.ndarray, std7: np.ndarray,
                 shuffle_buffer: int = 20000, seed: int = 123):
        super().__init__()
        self.paths = [Path(p) for p in parquet_paths]
        self.mean7 = mean7.astype(np.float32)
        self.std7  = std7.astype(np.float32)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)

    def _iter_worker(self, paths_subset: List[Path]):
        rng = np.random.RandomState(self.seed + (get_worker_info().id if get_worker_info() else 0))
        buf = []

        for p in paths_subset:
            df = pd.read_parquet(p)

            # sanity: colonne richieste
            for c in FEATURE_NAMES + TARGET_NAMES:
                if c not in df.columns:
                    raise KeyError(f"Colonna '{c}' mancante nel file {p}")

            X = df[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False)
            Y = df[TARGET_NAMES].to_numpy(dtype=np.float32, copy=False)

            # normalizza input con mean/std del train (streaming calcolate dal builder)
            Xn = (X - self.mean7[None,:]) / (self.std7[None,:] + 1e-6)

            # push nel buffer e yield con reservoir-like shuffle
            for i in range(Xn.shape[0]):
                item = (torch.from_numpy(Xn[i]), torch.from_numpy(Y[i]))
                buf.append(item)
                if len(buf) >= self.shuffle_buffer:
                    j = rng.randint(0, len(buf))
                    yield buf.pop(j)

            # svuota buffer
            rng.shuffle(buf)
            while buf:
                yield buf.pop()

            del df, X, Y, Xn

    def __iter__(self):
        info = get_worker_info()
        paths = list(self.paths)
        random.Random(self.seed + (info.id if info else 0)).shuffle(paths)
        if info is None:
            return self._iter_worker(paths)
        # split round-robin tra worker
        subset = [paths[i] for i in range(len(paths)) if i % info.num_workers == info.id]
        return self._iter_worker(subset)

# ======================
# Modello MLP
# ======================
class MLPImpact(nn.Module):
    def __init__(self, in_dim=7, hidden=128, layers=3, dropout=0.0):
        super().__init__()
        assert layers >= 1
        blocks = []
        d = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout > 0:
                blocks += [nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(d, 2)
        self.out_act = nn.Sigmoid()  # target (x_imp,y_imp) in [0,1]

    def forward(self, x):
        h = self.backbone(x)
        y = self.head(h)
        return self.out_act(y)

# ======================
# Helper I/O
# ======================
def list_train_parquet(compact_dir: Path) -> List[Path]:
    d = compact_dir / "train_compact"
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nessun parquet trovato in {d}")
    return files

def get_test_parquet(compact_dir: Path) -> Path:
    p = compact_dir / "test_compact" / "test.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Test parquet non trovato: {p}")
    return p

def load_stats(compact_dir: Path) -> Dict[str, Any]:
    sp = compact_dir / "stats" / "stats.json.gz"
    if not sp.exists():
        raise FileNotFoundError(f"Stats non trovate: {sp}")
    stats = load_json_gz(str(sp))
    # Verifica chiavi necessarie
    for k in ("mean7","std7","H","W"):
        if k not in stats:
            raise KeyError(f"Chiave '{k}' mancante in stats {sp}")
    return stats

# ======================
# Valutazione
# ======================
@torch.no_grad()
def evaluate(model: nn.Module, Xn: torch.Tensor, Yn: torch.Tensor, H: int, W: int, device) -> Dict[str, float]:
    model.eval()
    preds = []
    B = 4096
    for i in range(0, Xn.size(0), B):
        xb = Xn[i:i+B].to(device, non_blocking=True)
        yb = model(xb).cpu()
        preds.append(yb)
    yhat = torch.cat(preds, dim=0)
    y = Yn

    # errori su [0,1]
    mse = torch.mean((yhat - y)**2).item()
    mae = torch.mean(torch.abs(yhat - y)).item()

    # metriche in pixel
    Wm1, Hm1 = (W-1), (H-1)
    yhat_px = torch.stack([yhat[:,0]*Wm1, yhat[:,1]*Hm1], dim=1)
    y_px    = torch.stack([y[:,0]*Wm1,    y[:,1]*Hm1],    dim=1)

    mse_px = torch.mean((yhat_px - y_px)**2).item()
    mae_px = torch.mean(torch.abs(yhat_px - y_px)).item()
    rmse_px = math.sqrt(mse_px)
    mean_dist_px = torch.mean(torch.sqrt(torch.sum((yhat_px - y_px)**2, dim=1))).item()

    return dict(mse=mse, mae=mae, mse_px=mse_px, mae_px=mae_px, rmse_px=rmse_px, mean_dist_px=mean_dist_px)

# ======================
# Main
# ======================
def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compact_dir = Path(args.compact_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Stats globali
    stats = load_stats(compact_dir)
    mean7 = np.asarray(stats["mean7"], dtype=np.float32)
    std7  = np.asarray(stats["std7"],  dtype=np.float32)
    H, W  = int(stats["H"]), int(stats["W"])

    # Parquet train/test
    train_all = list_train_parquet(compact_dir)
    train_files = train_all[:max(1, min(args.train_shards, len(train_all)))]
    print(f"[DATA] Train shards: {len(train_files)} / {len(train_all)}")

    test_file = get_test_parquet(compact_dir)
    df_test = pd.read_parquet(test_file)

    # check colonne
    for c in FEATURE_NAMES + TARGET_NAMES:
        if c not in df_test.columns:
            raise KeyError(f"Colonna '{c}' mancante nel test set ({test_file}).")

    # prepara tensori test
    X_te = df_test[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False)
    Y_te = df_test[TARGET_NAMES].to_numpy(dtype=np.float32, copy=False)
    X_te_n = (X_te - mean7[None,:]) / (std7[None,:] + 1e-6)
    X_te_t = torch.from_numpy(X_te_n).contiguous()
    Y_te_t = torch.from_numpy(Y_te).contiguous()

    # Dataset/Dataloader train
    train_ds = ImpactIterable(train_files, mean7, std7, shuffle_buffer=args.shuffle_buffer, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        drop_last=True
    )

    # Modello/ottimizzazione
    model = MLPImpact(in_dim=7, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available() and not args.no_amp))
    loss_fn = nn.MSELoss()

    best_rmse_px = float("inf")
    best_path = out_dir / "impact_head_best.pt"

    # === CSV logging ===
    metrics_csv = out_dir / "stageA_metrics.csv"
    # header se non esiste
    if not metrics_csv.exists():
        pd.DataFrame([{
            "epoch": 0,
            "train_loss": np.nan,
            "test_mse": np.nan,
            "test_mae": np.nan,
            "test_mse_px": np.nan,
            "test_mae_px": np.nan,
            "test_rmse_px": np.nan,
            "test_meanDist_px": np.nan,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_shards": len(train_files),
            "epoch_time_sec": np.nan,
        }]).to_csv(metrics_csv, index=False)

    print(f"[INFO] Start training: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    for ep in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        run_loss = 0.0; seen = 0; step = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(torch.cuda.is_available() and not args.no_amp)):
                yhat = model(xb)
                loss = loss_fn(yhat, yb)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()

            run_loss += float(loss.detach()) * xb.size(0); seen += xb.size(0); step += 1
            if args.print_every and (step % args.print_every == 0):
                print(f"  [ep {ep:03d}] step {step:05d} | loss {run_loss/seen:.6f}")

        sched.step()
        train_loss = run_loss / max(1, seen)

        # Eval su test
        metrics = evaluate(model, X_te_t, Y_te_t, H, W, device)
        dt = time.time() - t0
        print(f"[EP {ep:03d}] train_loss={train_loss:.6f} | "
              f"test: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, "
              f"MSE_px={metrics['mse_px']:.3f}, MAE_px={metrics['mae_px']:.3f}, "
              f"RMSE_px={metrics['rmse_px']:.3f}, meanDist_px={metrics['mean_dist_px']:.3f} "
              f"| {dt:.1f}s")

        # === append CSV riga per epoca ===
        row = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "test_mse": float(metrics["mse"]),
            "test_mae": float(metrics["mae"]),
            "test_mse_px": float(metrics["mse_px"]),
            "test_mae_px": float(metrics["mae_px"]),
            "test_rmse_px": float(metrics["rmse_px"]),
            "test_meanDist_px": float(metrics["mean_dist_px"]),
            "lr": float(opt.param_groups[0]["lr"]),
            "batch_size": int(args.batch_size),
            "train_shards": int(len(train_files)),
            "epoch_time_sec": float(dt),
        }
        pd.DataFrame([row]).to_csv(metrics_csv, mode="a", header=False, index=False)

        # checkpoint best
        if metrics["rmse_px"] < best_rmse_px and np.isfinite(metrics["rmse_px"]):
            best_rmse_px = metrics["rmse_px"]
            payload = {
                "state_dict": model.state_dict(),
                "config": {
                    "in_dim": 7, "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
                    "mean7": stats["mean7"], "std7": stats["std7"], "H": H, "W": W,
                    "feature_names": FEATURE_NAMES, "target_names": TARGET_NAMES,
                },
                "optimizer": opt.state_dict(),
                "metrics": metrics,
                "epoch": ep,
            }
            torch.save(payload, best_path)
            print(f"  -> saved BEST to {best_path} (RMSE_px={best_rmse_px:.3f})")

    print("Training finito.")
    print(f"Best checkpoint: {best_path}")
    print(f"CSV metrics: {metrics_csv}")

if __name__ == "__main__":
    main()

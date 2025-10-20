#!/usr/bin/env python3
"""
Plot report per Stage A (impact head):
- Legge metrics CSV, seleziona epoca best con min test_rmse_px.
- Plotta loss vs epoca con barra verticale sull'epoca best.
- Plotta 3 metriche (MSE_px, RMSE_px, meanDist_px) in una canva con 3 righe, con barra verticale.
- Carica il checkpoint best, inferisce sul test e produce:
  * Spazio pixel: 2D istogrammi (reale vs pred), proiezioni 1D x e y (reale vs pred).
  * Spazio [0,1]: 2D istogrammi (reale vs pred), proiezioni 1D x e y (reale vs pred).

Dipendenze: torch, pandas, numpy, matplotlib, pyarrow
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath("/data/dataalice/dfuligno/ZDC_fastsim/src"))
from utility import seed_everything, load_json_gz

FEATURE_NAMES = ["E","vx","vy","vz","px","py","pz"]
TARGET_NAMES  = ["x_imp","y_imp"]

# ===== Model definition must match Stage A =====
class MLPImpact(nn.Module):
    def __init__(self, in_dim=7, hidden=128, layers=3, dropout=0.0):
        super().__init__()
        assert layers >= 1
        blocks = []
        d = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout and dropout > 0:
                blocks += [nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(d, 2)
        self.out_act = nn.Sigmoid()
    def forward(self, x):
        return self.out_act(self.head(self.backbone(x)))

def parse_args():
    ap = argparse.ArgumentParser(description="Plot report Stage A")
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/",
                    help="Cartella con train_compact/, test_compact/test.parquet, stats/stats.json.gz")
    ap.add_argument("--out-dir", default="stageA_Res",
                    help="Cartella di output per i PNG")
    ap.add_argument("--metrics-csv", default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_A/stageA_metrics.csv",
                    help="CSV prodotto dallo Stage A (stageA_metrics.csv)")
    ap.add_argument("--best-ckpt", default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_A/impact_head_best.pt",
                    help="Percorso al checkpoint best (impact_head_best.pt)")
    ap.add_argument("--bins-01", type=int, default=64, help="Bins per istogrammi in spazio [0,1]")
    ap.add_argument("--dpi", type=int, default=140)
    return ap.parse_args()

def find_best_epoch(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    # gestisci eventuali righe placeholder (epoch==0 con NaN)
    df = df[df["epoch"].notna()]
    df["epoch"] = df["epoch"].astype(int)
    best_row = df.loc[df["test_rmse_px"].idxmin()]
    return int(best_row["epoch"])

def plot_loss_and_metrics(csv_path: Path, best_epoch: int, out_dir: Path, dpi=140):
    df = pd.read_csv(csv_path)
    df = df[df["epoch"].notna()].copy()
    df["epoch"] = df["epoch"].astype(int)

    # 1) Loss vs epoca
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(df["epoch"], df["train_loss"])
    ax.axvline(best_epoch, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss (MSE on [0,1])")
    ax.set_title("Stage A — Train loss per epoch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "A_loss_vs_epoch.png", dpi=dpi)
    plt.close(fig)

    # 2) 3 metriche in una canva (3 righe)
    mets = [("test_mse_px", "Test MSE (pixel^2)"),
            ("test_rmse_px","Test RMSE (pixel)"),
            ("test_meanDist_px","Test mean distance (pixel)")]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
    for ax, (col, ylab) in zip(axes, mets):
        ax.plot(df["epoch"], df[col])
        ax.axvline(best_epoch, linestyle="--")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Stage A — Metrics per epoch (vertical line = best by RMSE_px)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_dir / "A_metrics_vs_epoch.png", dpi=dpi)
    plt.close(fig)

def load_best_and_predict(best_ckpt: Path, compact_dir: Path):
    # stats (mean/std/H/W)
    stats = load_json_gz(str(compact_dir / "stats" / "stats.json.gz"))
    mean7 = np.asarray(stats["mean7"], dtype=np.float32)
    std7  = np.asarray(stats["std7"], dtype=np.float32)
    H, W  = int(stats["H"]), int(stats["W"])

    # test set
    test_df = pd.read_parquet(compact_dir / "test_compact" / "test.parquet")
    X = test_df[FEATURE_NAMES].to_numpy(np.float32, copy=False)
    Y = test_df[TARGET_NAMES].to_numpy(np.float32, copy=False)  # in [0,1]

    Xn = (X - mean7[None,:]) / (std7[None,:] + 1e-6)
    xt = torch.from_numpy(Xn)

    # load model
    ckpt = torch.load(best_ckpt, map_location="cpu")
    cfg  = ckpt["config"]
    model = MLPImpact(in_dim=7, hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        yh = model(xt).cpu().numpy()  # [0,1]

    # pixel space
    x_real_px = Y[:,0] * (W - 1)
    y_real_px = Y[:,1] * (H - 1)
    x_pred_px = yh[:,0] * (W - 1)
    y_pred_px = yh[:,1] * (H - 1)

    return dict(
        Y01=Y, Yhat01=yh, H=H, W=W,
        x_real_px=x_real_px, y_real_px=y_real_px,
        x_pred_px=x_pred_px, y_pred_px=y_pred_px
    )

def plot_2d_hists(Y01, Yhat01, H, W, out_dir: Path, bins01=64, dpi=140):
    # ---- spazio [0,1] ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    # reale
    ax = axes[0]
    ax.hist2d(Y01[:,0], Y01[:,1], bins=bins01)
    ax.set_title("[0,1] — Real impact 2D hist")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # predetto
    ax = axes[1]
    ax.hist2d(Yhat01[:,0], Yhat01[:,1], bins=bins01)
    ax.set_title("[0,1] — Pred impact 2D hist")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_dir / "B_01_2D_hist_real_vs_pred.png", dpi=dpi)
    plt.close(fig)

def plot_2d_hists_pixels(x_real_px, y_real_px, x_pred_px, y_pred_px, H, W, out_dir: Path, dpi=140):
    # ---- spazio pixel ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    # reale
    ax = axes[0]
    ax.hist2d(x_real_px, y_real_px, bins=[W, H], range=[[0, W-1], [0, H-1]])
    ax.set_title("Pixel — Real impact 2D hist")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    # predetto
    ax = axes[1]
    ax.hist2d(x_pred_px, y_pred_px, bins=[W, H], range=[[0, W-1], [0, H-1]])
    ax.set_title("Pixel — Pred impact 2D hist")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.tight_layout()
    fig.savefig(out_dir / "C_px_2D_hist_real_vs_pred.png", dpi=140)
    plt.close(fig)

def plot_1d_projections_01(Y01, Yhat01, out_dir: Path, bins01=64, dpi=140):
    # x-proj
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.hist(Y01[:,0], bins=bins01, alpha=0.5, label="real", density=True)
    ax.hist(Yhat01[:,0], bins=bins01, alpha=0.5, label="pred", density=True)
    ax.set_title("[0,1] — X projection (real vs pred)")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "D_01_x_projection.png", dpi=dpi)
    plt.close(fig)

    # y-proj
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.hist(Y01[:,1], bins=bins01, alpha=0.5, label="real", density=True)
    ax.hist(Yhat01[:,1], bins=bins01, alpha=0.5, label="pred", density=True)
    ax.set_title("[0,1] — Y projection (real vs pred)")
    ax.set_xlabel("y")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "E_01_y_projection.png", dpi=dpi)
    plt.close(fig)

def plot_1d_projections_px(x_real_px, y_real_px, x_pred_px, y_pred_px, H, W, out_dir: Path, dpi=140):
    # x-proj pixel
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.hist(x_real_px, bins=W, range=(0, W-1), alpha=0.5, label="real", density=True)
    ax.hist(x_pred_px, bins=W, range=(0, W-1), alpha=0.5, label="pred", density=True)
    ax.set_title("Pixel — X projection (real vs pred)")
    ax.set_xlabel("x [px]"); ax.set_ylabel("density")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "F_px_x_projection.png", dpi=dpi)
    plt.close(fig)

    # y-proj pixel
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.hist(y_real_px, bins=H, range=(0, H-1), alpha=0.5, label="real", density=True)
    ax.hist(y_pred_px, bins=H, range=(0, H-1), alpha=0.5, label="pred", density=True)
    ax.set_title("Pixel — Y projection (real vs pred)")
    ax.set_xlabel("y [px]"); ax.set_ylabel("density")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "G_px_y_projection.png", dpi=dpi)
    plt.close(fig)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = Path(args.metrics_csv)
    best_ckpt   = Path(args.best_ckpt)
    compact_dir = Path(args.compact_dir)

    # 1) best epoch & plots epoch-wise
    best_epoch = find_best_epoch(metrics_csv)
    plot_loss_and_metrics(metrics_csv, best_epoch, out_dir, dpi=args.dpi)

    # 2) best checkpoint → predictions su test
    data = load_best_and_predict(best_ckpt, compact_dir)
    Y01     = data["Y01"]
    Yhat01  = data["Yhat01"]
    H, W    = data["H"], data["W"]
    x_r, y_r = data["x_real_px"], data["y_real_px"]
    x_p, y_p = data["x_pred_px"], data["y_pred_px"]

    # 3) canvas richieste
    plot_2d_hists(Y01, Yhat01, H, W, out_dir, bins01=args.bins_01, dpi=args.dpi)          # [0,1] 2D (2 plot affiancati)
    plot_2d_hists_pixels(x_r, y_r, x_p, y_p, H, W, out_dir, dpi=args.dpi)                 # pixel 2D (2 plot affiancati)
    plot_1d_projections_01(Y01, Yhat01, out_dir, bins01=args.bins_01, dpi=args.dpi)       # [0,1] proiezioni X/Y (2 plot)
    plot_1d_projections_px(x_r, y_r, x_p, y_p, H, W, out_dir, dpi=args.dpi)               # pixel proiezioni X/Y (2 plot)

    print("[DONE] Plots salvati in:", out_dir.resolve())

if __name__ == "__main__":
    main()

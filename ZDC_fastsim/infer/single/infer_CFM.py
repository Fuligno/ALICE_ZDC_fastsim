#!/usr/bin/env python3
# ============================================================
# vanilla_infer_eval.py — Inferenza & Valutazione CFM "vanilla"
# (matrici 44x44 originali: nessuna traslazione/normalizzazione)
#
# Output:
#  - grid_real_vs_gen_8x2.png
#  - entropy_hist_real_vs_gen.png
#  - occupancy_maps_real_vs_gen.png, occupancy_proj_{x,y}_overlay.png
#  - sum_per_image_hist_real_vs_gen.png
#  - sum_maps_real_vs_gen.png, sum_proj_{x,y}_overlay.png
#  - avg_gen_time.txt
#  - train_val_loss.png (se train_log.csv disponibile accanto al ckpt)
# ============================================================

from __future__ import annotations
import os, math, time, argparse, json, gzip, random
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------- Costanti --------------------
H = 44
W = 44
COND_COLS_FALLBACK = ["E","vx","vy","vz","px","py","pz"]  # solo fallback
EPS = 1e-12

# -------------------- Utils IO --------------------
def load_json_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Dataset test: matrici ORIGINALI --------------------
class FullMatrixDataset(torch.utils.data.Dataset):
    """
    Carica righe dal compact parquet & immagini (LMDB se presente, altrimenti shard .pkl).
    NON applica traslazioni o normalizzazioni: restituisce la matrice così com'è (float32).
    """
    def __init__(self, rows_meta: List[Dict[str, Any]], cond_cols: List[str]):
        super().__init__()
        self.cond_cols = cond_cols

        self.parquets: List[pd.DataFrame] = []
        self.global_to_local: List[Tuple[int,int]] = []  # (pid, rid)
        for meta in rows_meta:
            pq = pd.read_parquet(meta["parquet"])
            pid = len(self.parquets)
            self.parquets.append(pq)
            n = len(pq)
            for rid in range(n):
                self.global_to_local.append((pid, rid))

        self.per_pq_io = []
        for meta in rows_meta:
            self.per_pq_io.append({
                "lmdb": meta.get("lmdb", None),
                "shard_src": meta.get("shard_src", None),
                "image_col": meta.get("image_col", None),
            })
        self.pkl_cache: Dict[str, pd.DataFrame] = {}

    def __len__(self):
        return len(self.global_to_local)

    def _read_image_from_lmdb(self, lmdb_path: str, idx: int) -> np.ndarray:
        import lmdb
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            key = f"{idx:08d}".encode("ascii")
            buf = txn.get(key)
            if buf is None:
                env.close()
                raise KeyError(f"Idx {idx} non trovato in LMDB {lmdb_path}")
            arr = np.frombuffer(buf, dtype=np.float32).reshape(H, W)
        env.close()
        return arr

    def _read_image_from_pkl(self, shard_src: str, image_col: str, idx: int) -> np.ndarray:
        if shard_src not in self.pkl_cache:
            df = pd.read_pickle(shard_src)
            self.pkl_cache[shard_src] = df
        df = self.pkl_cache[shard_src]
        val = df.iloc[idx][image_col]
        arr = np.asarray(val)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape != (H,W):
            arr = arr.astype(np.float32, copy=False)
            if arr.shape != (H,W):
                raise ValueError(f"Image shape mismatch in {shard_src}[{idx}]: {arr.shape} != {(H,W)}")
        return arr.astype(np.float32, copy=False)

    def __getitem__(self, i: int):
        pid, rid = self.global_to_local[i]
        row = self.parquets[pid].iloc[rid]
        io = self.per_pq_io[pid]

        if io["lmdb"]:
            img_np = self._read_image_from_lmdb(io["lmdb"], rid)
        elif io["shard_src"] and io["image_col"]:
            img_np = self._read_image_from_pkl(io["shard_src"], io["image_col"], rid)
        else:
            raise RuntimeError("Né LMDB né shard_src disponibili.")

        x0 = torch.from_numpy(np.array(img_np, dtype=np.float32, copy=True)).unsqueeze(0)  # [1,H,W]
        cond_vals = [float(row[c]) for c in self.cond_cols]
        c = torch.tensor(cond_vals, dtype=torch.float32)  # [C]
        return x0, c

# -------------------- Modello: U-Net 2D (come nel training vanilla) --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(min(groups, out_ch), out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(min(groups, out_ch), out_ch), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class UNetCFM(nn.Module):
    def __init__(self, cond_dim: int, base: int = 32):
        super().__init__()
        self.c_proj = nn.Linear(cond_dim + 1, base)  # +1 per t
        self.enc1 = DoubleConv(1 + 1, base)          # +1 per canale cond
        self.down1 = nn.Conv2d(base, base*2, 3, stride=2, padding=1)
        self.enc2 = DoubleConv(base*2, base*2)
        self.down2 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1)
        self.mid  = DoubleConv(base*4, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
        self.out  = nn.Conv2d(base, 1, 1)

    def forward(self, x, t, c):
        B = x.size(0)
        tc = torch.cat([t.view(B,1), c], dim=1)
        c_embed = self.c_proj(tc).view(B, -1, 1, 1)
        c_map = c_embed[:, :1].repeat(1, 1, H, W)
        x_in = torch.cat([x, c_map], dim=1)
        h1 = self.enc1(x_in)
        h2 = self.enc2(self.down1(h1))
        h3 = self.mid(self.down2(h2))
        u2 = self.up2(h3)
        d2 = self.dec2(torch.cat([u2, h2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, h1], dim=1))
        out = self.out(d1)
        return out

# -------------------- Sampler Flow Matching (no post-normalization) --------------------
@torch.no_grad()
def make_time_grid(device: torch.device, steps: int, gamma: float):
    u = torch.linspace(0.0, 1.0, steps+1, device=device)
    return torch.pow(1.0 - u, gamma)  # t[0]=1 -> t[-1]=0

@torch.no_grad()
def sample_cfm(model: UNetCFM, c: torch.Tensor, steps: int = 200, sigma: float = 1.0,
               device=None, scheme: str="rk4_nu", time_gamma: float=2.5):
    if device is None: device = next(model.parameters()).device
    B = c.size(0)
    x = sigma * torch.randn(B,1,H,W, device=device)
    model.eval()

    if scheme.endswith("_nu"):
        t_grid = make_time_grid(device, steps, time_gamma)  # non uniforme
        for s in range(1, steps+1):
            t_hi = t_grid[s-1]; t_lo = t_grid[s]; dt = (t_hi - t_lo).item()
            t_hi_t  = torch.full((B,1), t_hi.item(), device=device)
            t_mid_t = torch.full((B,1), (0.5*(t_hi+t_lo)).item(), device=device)
            t_lo_t  = torch.full((B,1), t_lo.item(), device=device)
            if scheme == "heun_nu":
                v1 = model(x, t_hi_t, c)
                x_pred = x - dt * v1
                v2 = model(x_pred, t_lo_t, c)
                x = x - 0.5*dt*(v1+v2)
            elif scheme == "rk4_nu":
                k1 = model(x,            t_hi_t,  c)
                x2 = x - 0.5*dt*k1; k2 = model(x2, t_mid_t, c)
                x3 = x - 0.5*dt*k2; k3 = model(x3, t_mid_t, c)
                x4 = x - dt*k3;     k4 = model(x4, t_lo_t,  c)
                x = x - (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError("scheme non riconosciuto")
    else:
        dt = 1.0/steps
        for k in range(steps, 0, -1):
            t_k   = torch.full((B,1), k*dt, device=device)
            if scheme == "euler":
                v = model(x, t_k, c)
                x = x - dt * v
            elif scheme == "heun":
                t_km1 = torch.full((B,1), (k-1)*dt, device=device)
                v1 = model(x, t_k, c)
                x_pred = x - dt*v1
                v2 = model(x_pred, t_km1, c)
                x = x - 0.5*dt*(v1+v2)
            elif scheme == "rk4":
                t_h = torch.full((B,1), (k-0.5)*dt, device=device)
                t_km1 = torch.full((B,1), (k-1)*dt, device=device)
                k1 = model(x,            t_k,   c)
                x2 = x - 0.5*dt*k1; k2 = model(x2, t_h,   c)
                x3 = x - 0.5*dt*k2; k3 = model(x3, t_h,   c)
                x4 = x - dt*k3;     k4 = model(x4, t_km1, c)
                x = x - (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError("scheme non riconosciuto")
    return x  # raw, senza clamp/scale inversion qui

# -------------------- Plot helpers --------------------
def imshow_with_individual_colorbar(ax, img2d, title: str, cmap="viridis"):
    im = ax.imshow(img2d, origin="lower", interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(im, cax=cax)

def line_overlay(ax, xs, y_real, y_gen, title: str, xlabel: str):
    ax.plot(xs, y_real, label="Real")
    ax.plot(xs, y_gen,  label="Gen", linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel); ax.set_ylabel("counts")
    ax.legend(frameon=False)

# -------------------- Entropia normalizzata --------------------
def normalized_entropy_from_raw(x: np.ndarray) -> float:
    """x: [H,W] (raw, >=0 atteso). Normalizza internamente a somma=1 prima di calcolare H/Hmax."""
    p = x.astype(np.float64).ravel()
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0: 
        return 0.0
    p = p / s
    Hs = -(p * (np.log(p + EPS))).sum()
    Hmax = math.log(H*W)
    return float(Hs / (Hmax + EPS))

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Inferenza & Valutazione CFM Vanilla (matrici originali)")
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/",
                    help="Cartella con train_compact/, test_compact/, stats/manifest.json.gz")
    ap.add_argument("--ckpt", default="../../runs_cfm_vanilla/vanilla_cfm_best.pt",
                    help="Checkpoint .pt del CFM vanilla (contiene state_dict, cond_cols, intensity_scale)")
    ap.add_argument("--out-dir", default="../../runs_cfm_vanilla/",
                    help="Dove salvare figure e report")

    # sampling
    ap.add_argument("--steps", type=int, default=200, help="Passi integrazione")
    ap.add_argument("--sampler", choices=["euler","heun","rk4","heun_nu","rk4_nu"], default="rk4_nu")
    ap.add_argument("--time-gamma", type=float, default=2.5, help="γ per griglia non uniforme (solo *_nu)")
    ap.add_argument("--sigma", type=float, default=1.0, help="Std iniziale di x_1 ~ N(0, σ^2 I)")
    ap.add_argument("--no-clamp", action="store_true", help="Non clampare i valori generati a >=0")

    # valutazione
    ap.add_argument("--n-eval", type=int, default=1000, help="Numero di esempi casuali per statistiche")
    ap.add_argument("--num-sample-plot", type=int, default=8, help="Numero di coppie real/gen nella griglia")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch per generazione")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ------- Carica ckpt -------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cond_cols = ckpt.get("cond_cols", COND_COLS_FALLBACK)
    base_ch   = int(ckpt.get("base_ch", 32))
    inv_scale = float(ckpt.get("intensity_scale", 1.0))  # sarà usato per RIportare alla scala originale
    clip_min_train = float(ckpt.get("clip_min", 0.0))     # informativo

    print(f"[INFO] cond_cols: {cond_cols}")
    print(f"[INFO] intensity_scale (train) = {inv_scale} (in inferenza: dividiamo per questo valore)")
    print(f"[INFO] clip_min (train) = {clip_min_train}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetCFM(cond_dim=len(cond_cols), base=base_ch).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # ------- Leggi manifest test -------
    compact_dir = Path(args.compact_dir)
    manifest_path = compact_dir / "stats" / "manifest.json.gz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")
    manifest = load_json_gz(str(manifest_path))
    test_meta = manifest["test"]
    rows_test = [{
        "parquet": test_meta["parquet"],
        "lmdb": test_meta.get("lmdb", None),
        "shard_src": test_meta.get("shard_src", None),
        "image_col": test_meta.get("image_col", None),
    }]

    ds_te = FullMatrixDataset(rows_test, cond_cols)
    N_test = len(ds_te)
    print(f"[INFO] Test set size: {N_test}")

    # =============== 1) CANVA 8×2: Real vs Gen =================
    K = min(args.num_sample_plot, N_test)
    idxK = np.random.choice(N_test, size=K, replace=False)

    realsK = []
    condK  = []
    for i in idxK:
        x0, c = ds_te[i]
        realsK.append(x0.numpy()[0])  # [H,W]
        condK.append(c.numpy())
    condK_t = torch.tensor(np.stack(condK, axis=0), dtype=torch.float32, device=device)

    with torch.no_grad():
        x_gen = sample_cfm(model, condK_t, steps=args.steps, sigma=args.sigma,
                           device=device, scheme=args.sampler, time_gamma=args.time_gamma)
        # inverti la scala usata in train (moltiplicativa)
        if inv_scale != 1.0:
            x_gen = x_gen / inv_scale
        if not args.no_clamp:
            x_gen = torch.clamp(x_gen, min=0.0)
    gensK = x_gen.cpu().numpy()[:,0]

    fig, axes = plt.subplots(K, 2, figsize=(8, 2.4*K), constrained_layout=True)
    if K == 1: axes = np.array([axes])
    for r in range(K):
        imshow_with_individual_colorbar(axes[r,0], realsK[r], title=f"Real #{idxK[r]}")
        imshow_with_individual_colorbar(axes[r,1], gensK[r],  title=f"Gen  #{idxK[r]}")
    fig.suptitle("CFM Vanilla — Full matrix (Real vs Gen)", y=0.995, fontsize=12)
    fig.savefig(out_dir / "grid_real_vs_gen_8x2.png", dpi=150)
    plt.close(fig)

    # =============== 2) Statistiche su campione N =================
    N_eval = min(args.n_eval, N_test)
    idxN = np.random.choice(N_test, size=N_eval, replace=False)

    # Pre-estrai real & cond
    realsN = np.zeros((N_eval, H, W), dtype=np.float32)
    condN  = np.zeros((N_eval, len(cond_cols)), dtype=np.float32)
    for j, i in enumerate(idxN):
        x0, c = ds_te[i]
        realsN[j] = x0.numpy()[0]
        condN[j]  = c.numpy()

    # Generazione in batch + timing
    batch = args.batch_size
    genN  = np.zeros_like(realsN)
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for s in range(0, N_eval, batch):
            e = min(s+batch, N_eval)
            c_bt = torch.tensor(condN[s:e], dtype=torch.float32, device=device)
            xhat = sample_cfm(model, c_bt, steps=args.steps, sigma=args.sigma,
                              device=device, scheme=args.sampler, time_gamma=args.time_gamma)
            if inv_scale != 1.0:
                xhat = xhat / inv_scale
            if not args.no_clamp:
                xhat = torch.clamp(xhat, min=0.0)
            genN[s:e] = xhat.cpu().numpy()[:,0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    avg_time_per_item = (t1 - t0) / float(N_eval)
    with open(out_dir / "avg_gen_time.txt", "w") as f:
        f.write(f"Average generation time per item (sampling only): {avg_time_per_item:.6f} s\n")
    print(f"[INFO] Avg gen time/item: {avg_time_per_item:.6f} s")

    # ---- Entropia normalizzata (per immagine) ----
    ent_real = np.array([normalized_entropy_from_raw(realsN[i]) for i in range(N_eval)], dtype=np.float64)
    ent_gen  = np.array([normalized_entropy_from_raw(genN[i])   for i in range(N_eval)], dtype=np.float64)

    plt.figure(figsize=(6,4))
    bins = np.linspace(0, 1.0, 40)
    plt.hist(ent_real, bins=bins, alpha=0.6, label="Real", density=True)
    plt.hist(ent_gen,  bins=bins, alpha=0.6, label="Gen",  density=True, histtype="step", linewidth=1.8)
    plt.xlabel("Normalized entropy")
    plt.ylabel("Density")
    plt.title("Entropy distribution — Real vs Gen")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_hist_real_vs_gen.png", dpi=150)
    plt.close()

    # ---- OCCUPANCY (on/off > 0) ----
    thr = 0.0 + 1e-12
    occ_real = (realsN > thr).sum(axis=0).astype(np.float64)  # [H,W]
    occ_gen  = (genN   > thr).sum(axis=0).astype(np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    imshow_with_individual_colorbar(axes[0], occ_real, "Occupancy (Real)")
    imshow_with_individual_colorbar(axes[1], occ_gen,  "Occupancy (Gen)")
    fig.suptitle(f"Occupancy maps over {N_eval} samples", y=0.98, fontsize=12)
    fig.savefig(out_dir / "occupancy_maps_real_vs_gen.png", dpi=150)
    plt.close(fig)

    xs = np.arange(W); ys = np.arange(H)
    projx_real = occ_real.sum(axis=0); projx_gen = occ_gen.sum(axis=0)
    projy_real = occ_real.sum(axis=1); projy_gen = occ_gen.sum(axis=1)

    fig, ax = plt.subplots(figsize=(7,3.6))
    line_overlay(ax, xs, projx_real, projx_gen, title="Occupancy projection — X", xlabel="x (pixel)")
    fig.tight_layout(); fig.savefig(out_dir / "occupancy_proj_x_overlay.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,3.6))
    line_overlay(ax, ys, projy_real, projy_gen, title="Occupancy projection — Y", xlabel="y (pixel)")
    fig.tight_layout(); fig.savefig(out_dir / "occupancy_proj_y_overlay.png", dpi=150); plt.close(fig)

    # ---- SOMMA: (a) istogramma della somma per immagine ----
    sum_per_img_real = realsN.reshape(N_eval, -1).sum(axis=1)
    sum_per_img_gen  = genN.reshape(N_eval, -1).sum(axis=1)

    plt.figure(figsize=(6.4,4))
    # uso bins condivisi robusti
    all_sums = np.concatenate([sum_per_img_real, sum_per_img_gen])
    bins = np.histogram_bin_edges(all_sums, bins=50)
    plt.hist(sum_per_img_real, bins=bins, alpha=0.6, label="Real", density=True)
    plt.hist(sum_per_img_gen,  bins=bins, alpha=0.6, label="Gen",  density=True, histtype="step", linewidth=1.8)
    plt.xlabel("Total sum per image")
    plt.ylabel("Density")
    plt.title("Sum per image — Real vs Gen")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "sum_per_image_hist_real_vs_gen.png", dpi=150)
    plt.close()

    # ---- SOMMA: (b) sum map 2D + proiezioni ----
    sum_real = realsN.sum(axis=0).astype(np.float64)
    sum_gen  = genN.sum(axis=0).astype(np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    imshow_with_individual_colorbar(axes[0], sum_real, "Sum map (Real)")
    imshow_with_individual_colorbar(axes[1], sum_gen,  "Sum map (Gen)")
    fig.suptitle(f"Sum maps over {N_eval} samples", y=0.98, fontsize=12)
    fig.savefig(out_dir / "sum_maps_real_vs_gen.png", dpi=150)
    plt.close(fig)

    projx_real_sum = sum_real.sum(axis=0)
    projx_gen_sum  = sum_gen.sum(axis=0)
    projy_real_sum = sum_real.sum(axis=1)
    projy_gen_sum  = sum_gen.sum(axis=1)

    fig, ax = plt.subplots(figsize=(7,3.6))
    line_overlay(ax, xs, projx_real_sum, projx_gen_sum, title="Sum projection — X", xlabel="x (pixel)")
    fig.tight_layout(); fig.savefig(out_dir / "sum_proj_x_overlay.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,3.6))
    line_overlay(ax, ys, projy_real_sum, projy_gen_sum, title="Sum projection — Y", xlabel="y (pixel)")
    fig.tight_layout(); fig.savefig(out_dir / "sum_proj_y_overlay.png", dpi=150); plt.close(fig)

    # =============== 3) Curve di training dal CSV (se presente) ===============
    csv_guess = Path(args.ckpt).parent / "train_log.csv"
    if csv_guess.exists():
        df = pd.read_csv(csv_guess)
        best_ep = int(df.loc[df["val_loss"].idxmin(), "epoch"]) if "val_loss" in df.columns else int(df["epoch"].max())
        plt.figure(figsize=(7.2,4.2))
        if "train_loss" in df.columns:
            plt.plot(df["epoch"], df["train_loss"], label="Train loss")
        if "val_loss" in df.columns:
            plt.plot(df["epoch"], df["val_loss"],  label="Val loss")
        plt.axvline(best_ep, color="k", linestyle=":", linewidth=1.5, label=f"Best epoch = {best_ep}")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("CFM Vanilla — Training curves")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "train_val_loss.png", dpi=150)
        plt.close()
    else:
        print(f"[WARN] Log CSV non trovato in {csv_guess}. Salto il plot delle loss.")

    print(f"[DONE] Output salvati in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

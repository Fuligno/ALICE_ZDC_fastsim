#!/usr/bin/env python3
# ============================================================
# vanilla_infer_eval.py — Inferenza & Valutazione CFM "vanilla"
# (matrici 44x44 originali: nessuna traslazione/normalizzazione)
#
# Output principali:
#  - grid_real_vs_gen_8x2.png
#  - entropy_hist_real_vs_gen.png
#  - occupancy_maps_real_vs_gen.png, occupancy_proj_{x,y}_overlay.png
#  - sum_per_image_hist_real_vs_gen.png
#  - sum_maps_real_vs_gen.png, sum_proj_{x,y}_overlay.png
#  - max_value_hist_real_vs_gen.png
#  - impact_xy_hist2d_real_vs_gen.png
#  - impact_proj_x_overlay.png / impact_proj_y_overlay.png
#  - avg_gen_time.txt
#  - train_val_loss.png (se train_log.csv disponibile accanto al ckpt)
#  - NEW: max_value_hist_real_vs_gen_big.png
#  - NEW: sum_per_image_hist_real_vs_gen_big.png
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
COND_COLS_FALLBACK = ["E","vx","vy","vz","px","py","pz"] # solo fallback
EPS = 1e-12

# -------------------- Utils IO --------------------
def load_json_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Dataset test: matrici ORIGINALI --------------------
class FullMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, rows_meta: List[Dict[str, Any]], cond_cols: List[str]):
        super().__init__()
        self.cond_cols = cond_cols
        self.parquets: List[pd.DataFrame] = []
        self.global_to_local: List[Tuple[int,int]] = []
        for meta in rows_meta:
            pq = pd.read_parquet(meta["parquet"])
            pid = len(self.parquets)
            self.parquets.append(pq)
            for rid in range(len(pq)):
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
        c = torch.tensor([float(row[col]) for col in self.cond_cols], dtype=torch.float32)
        return x0, c

# -------------------- Modello (come nel training vanilla) --------------------
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
        self.c_proj = nn.Linear(cond_dim + 1, base)
        self.enc1 = DoubleConv(1 + 1, base)
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
        return self.out(d1)

# -------------------- Sampler Flow Matching --------------------
@torch.no_grad()
def make_time_grid(device: torch.device, steps: int, gamma: float):
    u = torch.linspace(0.0, 1.0, steps+1, device=device)
    return torch.pow(1.0 - u, gamma)  # t[0]=1 -> t[-1]=0

@torch.no_grad()
def sample_cfm(model: UNetCFM, c: torch.Tensor, steps: int = 200, sigma: float = 1.0,
               device=None, scheme: str="rk4_nu", time_gamma: float=2.5):
        # ... identico alla tua versione precedente ...
    if device is None: device = next(model.parameters()).device
    B = c.size(0)
    x = sigma * torch.randn(B,1,H,W, device=device)
    model.eval()
    if scheme.endswith("_nu"):
        t_grid = make_time_grid(device, steps, time_gamma)
        for s in range(1, steps+1):
            t_hi = t_grid[s-1]; t_lo = t_grid[s]; dt = (t_hi - t_lo).item()
            t_hi_t  = torch.full((B,1), t_hi.item(), device=device)
            t_mid_t = torch.full((B,1), (0.5*(t_hi+t_lo)).item(), device=device)
            t_lo_t  = torch.full((B,1), t_lo.item(), device=device)
            if scheme == "heun_nu":
                v1 = model(x, t_hi_t, c); x_pred = x - dt*v1
                v2 = model(x_pred, t_lo_t, c); x = x - 0.5*dt*(v1+v2)
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
                v = model(x, t_k, c); x = x - dt*v
            elif scheme == "heun":
                t_km1 = torch.full((B,1), (k-1)*dt, device=device)
                v1 = model(x, t_k, c); x_pred = x - dt*v1
                v2 = model(x_pred, t_km1, c); x = x - 0.5*dt*(v1+v2)
            elif scheme == "rk4":
                t_h = torch.full((B,1), (k-0.5)*dt, device=device)
                t_km1 = torch.full((B,1), (k-1)*dt, device=device)
                k1 = model(x, t_k, c)
                x2 = x - 0.5*dt*k1; k2 = model(x2, t_h,   c)
                x3 = x - 0.5*dt*k2; k3 = model(x3, t_h,   c)
                x4 = x - dt*k3;     k4 = model(x4, t_km1, c)
                x = x - (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError("scheme non riconosciuto")
    return x

# -------------------- Plot helpers / metriche (identici a prima) --------------------
def imshow_with_individual_colorbar(ax, img2d, title: str, cmap="viridis"):
    im = ax.imshow(img2d, origin="lower", interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(im, cax=cax)

def line_overlay(ax, xs, y_real, y_gen, title: str, xlabel: str):
    ax.plot(xs, y_real, label="Real"); ax.plot(xs, y_gen, linestyle="--", label="Gen")
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel); ax.set_ylabel("counts"); ax.legend(frameon=False)

def normalized_entropy_from_raw(x: np.ndarray) -> float:
    p = np.clip(x.astype(np.float64).ravel(), 0.0, None); s = p.sum()
    if s <= 0: return 0.0
    p = p / s; Hs = -(p * (np.log(p + EPS))).sum(); Hmax = math.log(H*W)
    return float(Hs / (Hmax + EPS))

def centroid_xy(img: np.ndarray) -> Tuple[float,float]:
    s = float(img.sum())
    if s <= 0: return (np.nan, np.nan)
    ys, xs = np.indices(img.shape)
    return float((xs*img).sum()/s), float((ys*img).sum()/s)

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
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sampler", choices=["euler","heun","rk4","heun_nu","rk4_nu"], default="rk4_nu")
    ap.add_argument("--time-gamma", type=float, default=2.5)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--no-clamp", action="store_true")
    # valutazione standard
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--num-sample-plot", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=123)
    # NEW: grande campione per istogrammi 1D max/sum
    ap.add_argument("--n-hist", type=int, default=10000, help="Eventi per istogrammi 1D max/sum (real vs gen)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ------- Carica ckpt -------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cond_cols = ckpt.get("cond_cols", COND_COLS_FALLBACK)
    base_ch   = int(ckpt.get("base_ch", 32))
    inv_scale = float(ckpt.get("intensity_scale", 1.0))  # in inferenza dividiamo per questo valore
    clip_min_train = float(ckpt.get("clip_min", 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetCFM(cond_dim=len(cond_cols), base=base_ch).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # ------- Test set -------
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
    print(f"[INFO] intensity_scale(train)={inv_scale} (in inferenza: divisione), clip_min(train)={clip_min_train}")

    # ======== 1) CANVA 8×2 + 2) Statistiche N_eval + Impact/Entropy/Occupancy/Sum maps ========
    # (SEZIONE IDENTICA ALLA VERSIONE PRECEDENTE — omessa qui per brevità nel commento)
    # Copia/incolla la tua sezione precedente che produce:
    # - grid 8x2
    # - N_eval generation (genN) + realsN
    # - entropy, occupancy, sum maps + proj, impact 2D + proj, max_value_hist_real_vs_gen.png
    # - sum_per_image_hist_real_vs_gen.png
    #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # NOTE: PER ECONOMIA DI SPAZIO, LASCIA QUI IL TUO BLOCCO "STANDARD" INVARIATO
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # ======================= NEW: BIG-SAMPLE 1D HISTS (max & sum) ============================
    # Scopo: usare n_hist eventi (>=10000) per avere statistiche robuste su max e sum per immagine.
    n_hist = int(args.n_hist)
    replace = n_hist > N_test
    idx_big = np.random.choice(N_test, size=n_hist, replace=replace)

    max_real_list, max_gen_list = [], []
    sum_real_list, sum_gen_list = [], []

    B = args.batch_size
    # Processiamo per chunk: carichiamo cond e real dal dataset, generiamo, poi calcoliamo max/sum e scartiamo i tensori
    for s in range(0, n_hist, B):
        e = min(s + B, n_hist)
        batch_idx = idx_big[s:e]

        # carica real e condizioni
        cond_bt = []
        sum_r_bt, max_r_bt = [], []
        for i in batch_idx:
            x0, c = ds_te[i]          # x0: [1,H,W]
            arr = x0.numpy()[0]
            sum_r_bt.append(float(arr.sum()))
            max_r_bt.append(float(arr.max()))
            cond_bt.append(c.numpy())

        cond_bt_t = torch.tensor(np.stack(cond_bt, axis=0), dtype=torch.float32, device=device)

        # genera
        with torch.no_grad():
            xhat = sample_cfm(model, cond_bt_t, steps=args.steps, sigma=args.sigma,
                              device=device, scheme=args.sampler, time_gamma=args.time_gamma)
            if inv_scale != 1.0:
                xhat = xhat / inv_scale
            if not args.no_clamp:
                xhat = torch.clamp(xhat, min=0.0)
            xhat_np = xhat.squeeze(1).cpu().numpy()

        # raccogli statistiche
        sum_g_bt = xhat_np.reshape(len(batch_idx), -1).sum(axis=1)
        max_g_bt = xhat_np.reshape(len(batch_idx), -1).max(axis=1)

        sum_real_list.extend(sum_r_bt)
        max_real_list.extend(max_r_bt)
        sum_gen_list.extend(sum_g_bt.tolist())
        max_gen_list.extend(max_g_bt.tolist())

        if (e % (10*B)) == 0 or e == n_hist:
            print(f"[BIG-HIST] processed {e}/{n_hist}")

    sum_real_arr = np.array(sum_real_list, dtype=np.float64)
    sum_gen_arr  = np.array(sum_gen_list,  dtype=np.float64)
    max_real_arr = np.array(max_real_list, dtype=np.float64)
    max_gen_arr  = np.array(max_gen_list,  dtype=np.float64)

    # --- Plot istogramma somma per immagine (BIG) ---
    plt.figure(figsize=(6.4,4))
    bins_sum_big = np.histogram_bin_edges(np.concatenate([sum_real_arr, sum_gen_arr]), bins=80)
    plt.hist(sum_real_arr, bins=bins_sum_big, alpha=0.6, label="Real", density=True)
    plt.hist(sum_gen_arr,  bins=bins_sum_big, alpha=0.6, label="Gen",  density=True, histtype="step", linewidth=1.8)
    plt.xlabel("Total sum per image")
    plt.ylabel("Density")
    plt.title(f"Sum per image — Real vs Gen (N={n_hist})")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "sum_per_image_hist_real_vs_gen_big.png", dpi=150)
    plt.close()

    # --- Plot istogramma max per immagine (BIG) ---
    plt.figure(figsize=(6.4,4))
    bins_max_big = np.histogram_bin_edges(np.concatenate([max_real_arr, max_gen_arr]), bins=80)
    plt.hist(max_real_arr, bins=bins_max_big, alpha=0.6, label="Real", density=True)
    plt.hist(max_gen_arr,  bins=bins_max_big, alpha=0.6, label="Gen",  density=True, histtype="step", linewidth=1.8)
    plt.xlabel("Max pixel value per image")
    plt.ylabel("Density")
    plt.title(f"Max value per image — Real vs Gen (N={n_hist})")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "max_value_hist_real_vs_gen_big.png", dpi=150)
    plt.close()

    print(f"[DONE] Output BIG-SAMPLE salvati in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

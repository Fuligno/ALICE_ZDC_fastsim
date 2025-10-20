#!/usr/bin/env python3
# ============================================================
# stageB_infer_eval.py — Inferenza & Valutazione Stage B (CFM)
# - Carica best checkpoint Stage B (UNetCFM)
# - Ricostruisce shape centrate & normalizzate dal test set
# - Genera shape via integrazione ODE (Euler) del vettore di flusso
# - Canva 8x2: real vs gen con colorbar indipendente per subplot
# - Campione di N=1000: tempo medio di generazione per elemento,
#   entropia normalizzata (istogrammi real vs gen),
#   istogrammi 2D di occupazione (on/off) + proiezioni x/y,
#   istogrammi 2D di somma (valori)     + proiezioni x/y
# - Plot train/val loss dal CSV con barra verticale all’epoca best
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
matplotlib.use("Agg")  # salva su file, no display
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------- Costanti --------------------
H = 44
W = 44
COND_COLS_DEFAULT = ["theta","phi","ux","uy","uz","E"]  # solo default; in pratica leggiamo dal ckpt
EPS = 1e-12

# -------------------- Utils IO --------------------
def load_json_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Warp/Subpixel shift --------------------
def fourier_shift_2d(img: torch.Tensor, tx: float, ty: float) -> torch.Tensor:
    """
    img: [H,W] float32; tx,ty in pixel (+x=destra, +y=giù)
    Traslazione sub-pixel via phase shift in Fourier.
    """
    Fimg = torch.fft.rfft2(img)
    yy = torch.fft.fftfreq(img.shape[0], d=1.0, device=img.device)   # H
    xx = torch.fft.rfftfreq(img.shape[1], d=1.0, device=img.device)  # W/2+1
    phase = torch.exp(-2j*math.pi*(yy[:,None]*ty + xx[None,:]*tx))
    shifted = torch.fft.irfft2(Fimg * phase, s=img.shape)
    return shifted

# -------------------- Dataset test: shape centrata & norm=1 --------------------
class ShapeCFMDataset(torch.utils.data.Dataset):
    """
    Carica righe dal compact parquet & immagini (LMDB se presente, altrimenti shard .pkl),
    centra e normalizza a somma=1 usando (x_imp,y_imp,T) oppure ricalcolando da immagine.
    Ritorna: x0:[1,H,W] (torch.float32), c:[C] (torch.float32)
    """
    def __init__(self, rows_meta: List[Dict[str, Any]], cond_cols: List[str], recompute_from_image: bool=False):
        super().__init__()
        self.cond_cols = cond_cols
        self.recompute = recompute_from_image

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

        # immagine
        if io["lmdb"]:
            img_np = self._read_image_from_lmdb(io["lmdb"], rid)
        elif io["shard_src"] and io["image_col"]:
            img_np = self._read_image_from_pkl(io["shard_src"], io["image_col"], rid)
        else:
            raise RuntimeError("Né LMDB né shard_src disponibili.")

        # centro+normalizzazione
        if self.recompute:
            total = float(img_np.sum())
            if total <= 0:
                cx_pix = (W-1)/2.0; cy_pix = (H-1)/2.0
            else:
                ys, xs = np.indices((H,W))
                cx_pix = float((xs*img_np).sum()/(total+EPS))
                cy_pix = float((ys*img_np).sum()/(total+EPS))
        else:
            cx01 = float(row["x_imp"]); cy01 = float(row["y_imp"])
            cx_pix = cx01 * (W-1.0)
            cy_pix = cy01 * (H-1.0)
            total = float(row["T"])

        tx = (W-1)/2.0 - cx_pix
        ty = (H-1)/2.0 - cy_pix

        img_t = torch.from_numpy(img_np)
        img_c = fourier_shift_2d(img_t, tx=tx, ty=ty)
        img_c = torch.clamp(img_c, min=0.0)
        if total > 0:
            img_c = img_c / float(total)
        s = img_c.sum().item()
        if s > 0:
            img_c = img_c / s

        x0 = img_c.unsqueeze(0).to(torch.float32)  # [1,H,W]

        cond_vals = [float(row[c]) for c in self.cond_cols]
        c = torch.tensor(cond_vals, dtype=torch.float32)  # [C]

        return x0, c

# -------------------- Modello: U-Net 2D minimal --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class UNetCFM(nn.Module):
    def __init__(self, cond_dim: int, base: int = 32):
        super().__init__()
        self.c_proj = nn.Linear(cond_dim+1, base)  # +1 per t
        self.enc1 = DoubleConv(1+1, base)          # +1 per c_embed (come canale)
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
        c_embed = self.c_proj(tc).view(B,-1,1,1)
        c_map = c_embed.repeat(1,1,H,W)
        x_in = torch.cat([x, c_map[:, :1]], dim=1)
        h1 = self.enc1(x_in)
        h2 = self.enc2(self.down1(h1))
        h3 = self.mid(self.down2(h2))
        u2 = self.up2(h3)
        d2 = self.dec2(torch.cat([u2, h2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, h1], dim=1))
        out = self.out(d1)
        return out

# -------------------- Sampling per Flow Matching (Euler) --------------------
@torch.no_grad()
def _postprocess_shape(x: torch.Tensor) -> torch.Tensor:
    """ clamp>=0 e normalizza a somma=1 per ogni item """
    B = x.size(0)
    x = torch.clamp(x, min=0.0)
    s = x.flatten(1).sum(dim=1).view(B,1,1,1) + 1e-12
    return x / s
def _make_time_grid(device: torch.device, steps: int, gamma: float) -> torch.Tensor:
    """
    Ritorna t[0..steps] con t[0]=1, t[steps]=0 e passi più fitti vicino a 0 se gamma>1:
      t(u) = (1-u)^gamma, u = linspace(0,1,steps+1)
    """
    u = torch.linspace(0.0, 1.0, steps+1, device=device)
    t = torch.pow(1.0 - u, gamma)
    return t  # shape [steps+1], decrescente da 1 -> 0

@torch.no_grad()
def sample_cfm_euler(model, cond, steps: int=50, sigma: float=1.0, device=None):
    """Euler esplicito su dx/dt = v(x,t,c), t: 1->0"""
    if device is None: device = next(model.parameters()).device
    B = cond.size(0)
    x = sigma * torch.randn(B,1,H,W, device=device)
    model.eval()
    dt = 1.0 / steps
    for k in range(steps, 0, -1):
        t_k = torch.full((B,1), k*dt, device=device)
        v = model(x, t_k, cond)
        x = x - dt * v
    return _postprocess_shape(x)

@torch.no_grad()
def sample_cfm_heun(model, cond, steps: int=120, sigma: float=1.0, device=None):
    """Heun (RK2): predictor-corrector"""
    if device is None: device = next(model.parameters()).device
    B = cond.size(0)
    x = sigma * torch.randn(B,1,H,W, device=device)
    model.eval()
    dt = 1.0 / steps
    for k in range(steps, 0, -1):
        t_k   = torch.full((B,1), k*dt, device=device)
        t_km1 = torch.full((B,1), (k-1)*dt, device=device)
        v1 = model(x, t_k, cond)
        x_pred = x - dt * v1
        v2 = model(x_pred, t_km1, cond)
        x = x - 0.5*dt*(v1 + v2)
    return _postprocess_shape(x)

@torch.no_grad()
def sample_cfm_rk4(model, cond, steps: int=80, sigma: float=1.0, device=None):
    """RK4 classico"""
    if device is None: device = next(model.parameters()).device
    B = cond.size(0)
    x = sigma * torch.randn(B,1,H,W, device=device)
    model.eval()
    dt = 1.0 / steps
    for k in range(steps, 0, -1):
        t_k   = torch.full((B,1), k*dt, device=device)
        t_h   = torch.full((B,1), (k-0.5)*dt, device=device)
        t_km1 = torch.full((B,1), (k-1)*dt, device=device)

        k1 = model(x,              t_k,   cond)
        x2 = x - 0.5*dt*k1
        k2 = model(x2,             t_h,   cond)
        x3 = x - 0.5*dt*k2
        k3 = model(x3,             t_h,   cond)
        x4 = x - dt*k3
        k4 = model(x4,             t_km1, cond)

        x = x - (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return _postprocess_shape(x)
@torch.no_grad()
def sample_cfm_heun_nu(model, cond, steps=120, gamma: float=2.5, sigma: float=1.0, device=None):
    """
    Heun (RK2) con griglia non uniforme t = (1-u)^gamma.
    Integrazione corretta dagli intervalli [t_hi=t[s-1], t_lo=t[s]] con dt = t_hi - t_lo > 0.
    """
    if device is None: device = next(model.parameters()).device
    B = cond.size(0)
    x = sigma * torch.randn(B,1,H,W, device=device)
    model.eval()

    t_grid = _make_time_grid(device, steps, gamma)  # t[0]=1 ... t[steps]=0

    for s in range(1, steps+1):
        t_hi = t_grid[s-1]           # ~ parte alta dell'intervallo (vicino a 1 all'inizio)
        t_lo = t_grid[s]             # ~ parte bassa (verso 0)
        dt = (t_hi - t_lo).item()    # positivo

        t_hi_t = torch.full((B,1), t_hi.item(), device=device)
        t_lo_t = torch.full((B,1), t_lo.item(), device=device)

        v1 = model(x, t_hi_t, cond)                  # pendenza a t_hi
        x_pred = x - dt * v1                         # predictor (Euler)
        v2 = model(x_pred, t_lo_t, cond)             # pendenza a t_lo
        x = x - 0.5 * dt * (v1 + v2)                 # corrector

    # postprocess
    x = torch.clamp(x, min=0.0)
    s = x.flatten(1).sum(dim=1).view(B,1,1,1) + 1e-12
    return x / s

@torch.no_grad()
def sample_cfm_rk4_nu(model, cond, steps=80, gamma: float=2.5, sigma: float=1.0, device=None):
    """
    RK4 classico con griglia non uniforme t = (1-u)^gamma.
    Usa l'intervallo [t_hi=t[s-1], t_lo=t[s]] e t_mid = (t_hi + t_lo)/2.
    """
    if device is None: device = next(model.parameters()).device
    B = cond.size(0)
    x = sigma * torch.randn(B,1,H,W, device=device)
    model.eval()

    t_grid = _make_time_grid(device, steps, gamma)  # t[0]=1 ... t[steps]=0

    for s in range(1, steps+1):
        t_hi = t_grid[s-1]
        t_lo = t_grid[s]
        dt = (t_hi - t_lo).item()        # positivo
        t_mid = 0.5 * (t_hi + t_lo)

        t_hi_t  = torch.full((B,1), t_hi.item(),  device=device)
        t_mid_t = torch.full((B,1), t_mid.item(), device=device)
        t_lo_t  = torch.full((B,1), t_lo.item(),  device=device)

        k1 = model(x,              t_hi_t,  cond)
        x2 = x - 0.5*dt*k1;  k2 = model(x2, t_mid_t, cond)
        x3 = x - 0.5*dt*k2;  k3 = model(x3, t_mid_t, cond)
        x4 = x - dt*k3;      k4 = model(x4, t_lo_t,  cond)

        x = x - (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # postprocess
    x = torch.clamp(x, min=0.0)
    s = x.flatten(1).sum(dim=1).view(B,1,1,1) + 1e-12
    return x / s


@torch.no_grad()
def sample_cfm(model, cond, sampler: str="heun", steps: int=120, sigma: float=1.0, device=None, time_gamma: float=2.5):
    if sampler == "euler":
        return sample_cfm_euler(model, cond, steps=steps, sigma=sigma, device=device)
    elif sampler == "rk4":
        return sample_cfm_rk4(model, cond, steps=steps, sigma=sigma, device=device)
    elif sampler == "heun_nu":
        return sample_cfm_heun_nu(model, cond, steps=steps, gamma=time_gamma, sigma=sigma, device=device)
    elif sampler == "rk4_nu":
        return sample_cfm_rk4_nu(model, cond, steps=steps, gamma=time_gamma, sigma=sigma, device=device)
    else:  # "heun"
        return sample_cfm_heun(model, cond, steps=steps, sigma=sigma, device=device)


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
def normalized_entropy(x: np.ndarray) -> float:
    """x: [H,W] >=0, somma ~1."""
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
    ap = argparse.ArgumentParser(description="Inferenza & Valutazione Stage B (CFM)")
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/", help="Cartella con train_compact/, test_compact/, stats/manifest.json.gz")
    ap.add_argument("--ckpt", default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_B/stageB_cfm_best.pt", help="Checkpoint .pt di Stage B (contiene state_dict e cond_cols)")
    ap.add_argument("--out-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/infer/stageB_Res", help="Dove salvare figure e report")
    ap.add_argument("--n-eval", type=int, default=1000, help="Numero di esempi casuali per statistiche")
    ap.add_argument("--num-sample-plot", type=int, default=8, help="Righe della canva (8 real/gen)")
    ap.add_argument("--ode-steps", type=int, default=500, help="Passi Euler per integrazione FM")
    ap.add_argument("--sampler",
                choices=["euler","heun","rk4","heun_nu","rk4_nu"],
                default="rk4_nu",
                help="Integratore ODE: euler | heun (RK2) | rk4 | heun_nu (griglia non uniforme) | rk4_nu")
    ap.add_argument("--time-gamma", type=float, default=3.0,
                help="Esponente γ della griglia non uniforme t=(1-u)^γ (usato da *_nu)")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch per generazione nelle statistiche")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--recompute-from-image", action="store_true", help="Ricalcola centro+T dall’immagine")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ------- Carica ckpt -------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cond_cols = ckpt.get("cond_cols", COND_COLS_DEFAULT)
    print(f"[INFO] cond_cols dal ckpt: {cond_cols}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetCFM(cond_dim=len(cond_cols), base=32).to(device)
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

    ds_te = ShapeCFMDataset(rows_test, cond_cols, recompute_from_image=args.recompute_from_image)
    N_test = len(ds_te)
    print(f"[INFO] Test set size: {N_test}")

    # ------------------------------------------------------------
    # 1) CANVA 8x2 — real vs gen con colorbar indipendente
    # ------------------------------------------------------------
    K = min(args.num_sample_plot, N_test)
    idx8 = np.random.choice(N_test, size=K, replace=False)
    reals_8 = []
    gens_8  = []
    cond_8  = []

    for i in idx8:
        x0, c = ds_te[i]
        reals_8.append(x0.numpy()[0])  # [H,W]
        cond_8.append(c.numpy())

    cond_8_t = torch.tensor(np.stack(cond_8, axis=0), dtype=torch.float32, device=device)
    with torch.no_grad():
        gen_8_t = sample_cfm(model, cond_8_t, sampler=args.sampler, steps=args.ode_steps, sigma=1.0, device=device, time_gamma=args.time_gamma)
    gens_8 = gen_8_t.cpu().numpy()[:,0]  # [K,H,W]

    # Plot 8x2
    fig, axes = plt.subplots(K, 2, figsize=(8, 2.4*K), constrained_layout=True)
    if K == 1:
        axes = np.array([axes])  # shape (1,2)
    for r in range(K):
        imshow_with_individual_colorbar(axes[r,0], reals_8[r], title=f"Real #{idx8[r]}")
        imshow_with_individual_colorbar(axes[r,1], gens_8[r],  title=f"Gen  #{idx8[r]}")
    fig.suptitle("Stage B — Shape centrata: Real vs Gen", y=0.995, fontsize=12)
    fig.savefig(out_dir / "grid_real_vs_gen_8x2.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------
    # 2) Statistiche su campione N (default 1000)
    #    - tempo medio di generazione per elemento (solo sampling)
    #    - entropia normalizzata (istogrammi overlay)
    #    - istogrammi 2D occupazione + proiezioni x/y
    #    - istogrammi 2D somma      + proiezioni x/y
    # ------------------------------------------------------------
    N_eval = min(args.n_eval, N_test)
    idxN = np.random.choice(N_test, size=N_eval, replace=False)
    # Pre-estrai real & cond (CPU)
    realsN = np.zeros((N_eval, H, W), dtype=np.float32)
    condN  = np.zeros((N_eval, len(cond_cols)), dtype=np.float32)
    for j, i in enumerate(idxN):
        x0, c = ds_te[i]
        realsN[j] = x0.numpy()[0]
        condN[j]  = c.numpy()

    # Generazione in batch + timing (solo sampling)
    batch = args.batch_size
    genN  = np.zeros_like(realsN)
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for s in range(0, N_eval, batch):
            e = min(s+batch, N_eval)
            c_bt = torch.tensor(condN[s:e], dtype=torch.float32, device=device)
            xhat = sample_cfm(model, c_bt, sampler=args.sampler, steps=args.ode_steps, sigma=1.0, device=device, time_gamma=args.time_gamma)
            genN[s:e] = xhat.cpu().numpy()[:,0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    avg_time_per_item = (t1 - t0) / float(N_eval)
    with open(out_dir / "avg_gen_time.txt", "w") as f:
        f.write(f"Average generation time per item (sampling only): {avg_time_per_item:.6f} s\n")
    print(f"[INFO] Avg gen time/item: {avg_time_per_item:.6f} s")

    # ---- Entropia normalizzata ----
    ent_real = np.array([normalized_entropy(realsN[i]) for i in range(N_eval)], dtype=np.float64)
    ent_gen  = np.array([normalized_entropy(genN[i])   for i in range(N_eval)], dtype=np.float64)

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

    # ---- Istogrammi 2D: OCCUPAZIONE (on/off >0) ----
    thr = 0.0 + 1e-12
    occ_real = (realsN > thr).sum(axis=0).astype(np.float64)  # [H,W]
    occ_gen  = (genN   > thr).sum(axis=0).astype(np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    imshow_with_individual_colorbar(axes[0], occ_real, "Occupancy (Real)")
    imshow_with_individual_colorbar(axes[1], occ_gen,  "Occupancy (Gen)")
    fig.suptitle(f"Occupancy maps over {N_eval} samples", y=0.98, fontsize=12)
    fig.savefig(out_dir / "occupancy_maps_real_vs_gen.png", dpi=150)
    plt.close(fig)

    # Proiezioni x/y (overlay real vs gen)
    projx_real = occ_real.sum(axis=0)  # [W]
    projx_gen  = occ_gen.sum(axis=0)
    projy_real = occ_real.sum(axis=1)  # [H]
    projy_gen  = occ_gen.sum(axis=1)

    xs = np.arange(W); ys = np.arange(H)

    fig, ax = plt.subplots(figsize=(7,3.6))
    line_overlay(ax, xs, projx_real, projx_gen, title="Occupancy projection — X", xlabel="x (pixel)")
    fig.tight_layout(); fig.savefig(out_dir / "occupancy_proj_x_overlay.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,3.6))
    line_overlay(ax, ys, projy_real, projy_gen, title="Occupancy projection — Y", xlabel="y (pixel)")
    fig.tight_layout(); fig.savefig(out_dir / "occupancy_proj_y_overlay.png", dpi=150); plt.close(fig)

    # ---- Istogrammi 2D: SOMMA (riempi col valore del pixel) ----
    sum_real = realsN.sum(axis=0).astype(np.float64)  # [H,W]
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

    # ------------------------------------------------------------
    # 3) Curve di training dal CSV + barra su epoca best
    # ------------------------------------------------------------
    csv_guess = Path(args.ckpt).parent / "stageB_train_log.csv"
    csv_path = csv_guess if csv_guess.exists() else (Path(args.compact_dir) / "stageB_train_log.csv")
    if not csv_path.exists():
        print(f"[WARN] Log CSV non trovato in {csv_guess}. Salto il plot delle loss.")
    else:
        df = pd.read_csv(csv_path)
        # best = min della val_loss
        if "val_loss" in df.columns:
            best_idx = int(df["val_loss"].idxmin())
            best_ep  = int(df.loc[best_idx, "epoch"])
        else:
            best_ep = int(df["epoch"].max())

        plt.figure(figsize=(7.2,4.2))
        if "train_loss" in df.columns:
            plt.plot(df["epoch"], df["train_loss"], label="Train loss")
        if "val_loss" in df.columns:
            plt.plot(df["epoch"], df["val_loss"],  label="Val loss")
        plt.axvline(best_ep, color="k", linestyle=":", linewidth=1.5, label=f"Best epoch = {best_ep}")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Stage B — Training curves")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "train_val_loss.png", dpi=150)
        plt.close()

    print(f"[DONE] Output salvati in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

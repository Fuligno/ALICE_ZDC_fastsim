#!/usr/bin/env python3
# ============================================================
# stageB_ddpm_infer_eval.py — Inferenza & Valutazione Stage B (DDPM)
# - Carica checkpoint DDPM (usa EMA se presente)
# - Ricostruisce shape centrate & normalizzate dal test set
# - Sampler: DDPM (ancestral) o DDIM (deterministico) con sotto-campionamento
# - Canva 8x2: real vs gen con colorbar indipendente
# - Campione di N: tempo medio, entropia, occupancy/sum maps + proiezioni
# - Plot train/val loss dal CSV con barra all’epoca best (robusto)
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
COND_COLS_DEFAULT = ["theta","phi","ux","uy","uz","E"]
EPS = 1e-12

# -------------------- Utils IO --------------------
def load_json_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Warp/Subpixel shift --------------------
def fourier_shift_2d(img: torch.Tensor, tx: float, ty: float) -> torch.Tensor:
    Fimg = torch.fft.rfft2(img)
    yy = torch.fft.fftfreq(img.shape[0], d=1.0, device=img.device)
    xx = torch.fft.rfftfreq(img.shape[1], d=1.0, device=img.device)
    phase = torch.exp(-2j*math.pi*(yy[:,None]*ty + xx[None,:]*tx))
    shifted = torch.fft.irfft2(Fimg * phase, s=img.shape)
    return shifted

# -------------------- Dataset test (come training DDPM) --------------------
class ShapeDDPMDataset(torch.utils.data.Dataset):
    def __init__(self, rows_meta: List[Dict[str, Any]], cond_cols: List[str], recompute_from_image: bool=False):
        super().__init__()
        self.cond_cols = cond_cols
        self.recompute = recompute_from_image
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

    def __len__(self): return len(self.global_to_local)

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
                raise ValueError(f"Image shape mismatch {arr.shape} != {(H,W)}")
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

        # centra + normalizza a somma=1
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
            cx_pix = cx01 * (W-1.0); cy_pix = cy01 * (H-1.0)
            total = float(row["T"])

        tx = (W-1)/2.0 - cx_pix; ty = (H-1)/2.0 - cy_pix
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
        c = torch.tensor(cond_vals, dtype=torch.float32)
        return x0, c

# -------------------- Modello: UNetDDPM (deve combaciare col training) --------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int): super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0)/(half-1)))
        args = t.float() * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class UNetDDPM(nn.Module):
    def __init__(self, cond_dim: int, base: int=32, tdim: int=64):
        super().__init__()
        self.t_emb = SinusoidalPosEmb(tdim)
        self.t_mlp = nn.Sequential(nn.Linear(tdim, base), nn.SiLU(), nn.Linear(base, base))
        self.c_proj = nn.Linear(cond_dim, base)

        self.enc1 = DoubleConv(1+1, base)
        self.down1 = nn.Conv2d(base, base*2, 3, stride=2, padding=1)
        self.enc2 = DoubleConv(base*2, base*2)
        self.down2 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1)
        self.mid  = DoubleConv(base*4, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
        self.out  = nn.Conv2d(base, 1, 1)

    def forward(self, x, t01, c):
        B = x.size(0)
        t_emb = self.t_mlp(self.t_emb(t01.view(B,1)))
        c_emb = self.c_proj(c)
        fuse = (t_emb + c_emb).view(B,-1,1,1).repeat(1,1,H,W)
        xc = torch.cat([x, fuse[:, :1]], dim=1)
        h1 = self.enc1(xc)
        h2 = self.enc2(self.down1(h1))
        h3 = self.mid(self.down2(h2))
        u2 = self.up2(h3)
        d2 = self.dec2(torch.cat([u2, h2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, h1], dim=1))
        eps = self.out(d1)
        return eps

# -------------------- Schedules & Samplers --------------------
def prepare_schedule_from_ckpt(ckpt: dict, device: torch.device):
    """Ritorna dict con betas, alphas, alphas_bar e vari precomputi su device."""
    betas = torch.tensor(ckpt["betas"], dtype=torch.float32, device=device)  # [T]
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sqrt_ab = torch.sqrt(alphas_bar)
    sqrt_1mab = torch.sqrt(1.0 - alphas_bar)
    sqrt_a = torch.sqrt(alphas)
    one_over_sqrt_a = 1.0 / sqrt_a
    # posterior variance (Ho et al.)
    T = betas.numel()
    alphas_bar_prev = torch.cat([torch.tensor([1.0], device=device), alphas_bar[:-1]], dim=0)
    posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar + 1e-12)
    posterior_log_var_clipped = torch.log(torch.clamp(posterior_var, min=1e-20))
    return dict(
        betas=betas, alphas=alphas, alphas_bar=alphas_bar,
        sqrt_ab=sqrt_ab, sqrt_1mab=sqrt_1mab, sqrt_a=sqrt_a,
        one_over_sqrt_a=one_over_sqrt_a,
        alphas_bar_prev=alphas_bar_prev,
        posterior_var=posterior_var,
        posterior_log_var_clipped=posterior_log_var_clipped
    )

def make_t_subset(T: int, steps: int) -> np.ndarray:
    """
    Sotto-campiona gli indici t in [T-1..0] in modo uniforme.
    Include sempre 0 e T-1. Esempio: T=1000, steps=50 → 50 indici decrescenti.
    """
    steps = int(steps)
    steps = max(2, min(steps, T))
    ts = np.linspace(T-1, 0, steps, dtype=np.float64)
    ts = np.round(ts).astype(np.int64)
    ts = np.unique(ts)             # crescente
    ts = ts[::-1]                  # decrescente
    if ts[-1] != 0: ts = np.concatenate([ts, [0]])
    return ts

@torch.no_grad()
def ddpm_sample(model, cond, sched, steps: int, device, sigma_scale: float=1.0):
    """
    DDPM ancestrale con sotto-campionamento: segue la formula del posterior mean + var al sotto-step t->t_next.
    sigma_scale = 1.0 (default DDPM). Metti 0.0 per DDIM-like (ma allora usa la funzione ddim_sample).
    """
    B = cond.size(0); T = sched["betas"].numel()
    x = torch.randn(B,1,H,W, device=device)
    t_subset = make_t_subset(T, steps)  # es. [999, 979, ..., 0]
    for idx, t in enumerate(t_subset):
        t_t = torch.full((B,1), float(t)/(T-1), device=device)  # normalizzato [0,1]
        eps = model(x, t_t, cond)
        # predizione x0
        sqrt_ab_t = sched["sqrt_ab"][t]; sqrt_1mab_t = sched["sqrt_1mab"][t]
        x0_pred = (x - sqrt_1mab_t * eps) / (sqrt_ab_t + 1e-12)

        # posterior mean per step t -> t' (successivo nella subset)
        if t == 0:
            x = x0_pred
            break
        t_next = t_subset[idx+1]  # è < t
        # ricostruisco i coefficienti corrispondenti a t e t_next
        a_t = sched["alphas"][t]; ab_t = sched["alphas_bar"][t]
        ab_next = sched["alphas_bar"][t_next]
        beta_t = sched["betas"][t]

        # Mean e var (vedi Ho et al. – al sotto-campionamento si usa la stessa formula con ab_next)
        coef1 = torch.sqrt(ab_next) * beta_t / (1.0 - ab_t + 1e-12)
        coef2 = torch.sqrt(a_t) * (1.0 - ab_next) / (1.0 - ab_t + 1e-12)
        mean = coef1.view(1,1,1,1) * x0_pred + coef2.view(1,1,1,1) * x

        var = (1.0 - ab_next) / (1.0 - ab_t + 1e-12) * beta_t
        std = sigma_scale * torch.sqrt(torch.clamp(var, min=1e-20))
        noise = torch.randn_like(x) if t_next > 0 else torch.zeros_like(x)
        x = mean + std.view(1,1,1,1) * noise

    # clamp & renormalize (shape)
    x = torch.clamp(x, min=0.0)
    s = x.flatten(1).sum(dim=1).view(B,1,1,1) + 1e-12
    return x / s

@torch.no_grad()
def ddim_sample(model, cond, sched, steps: int, device, eta: float=0.0):
    """
    DDIM deterministico (eta=0) o stocastico (eta>0), con sotto-campionamento uniforme.
    """
    B = cond.size(0); T = sched["betas"].numel()
    x = torch.randn(B,1,H,W, device=device)
    t_subset = make_t_subset(T, steps)  # [T-1 ... 0]
    for idx, t in enumerate(t_subset):
        t_t = torch.full((B,1), float(t)/(T-1), device=device)
        eps = model(x, t_t, cond)
        ab_t = sched["alphas_bar"][t]
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_1mab_t = torch.sqrt(1.0 - ab_t)

        x0_pred = (x - sqrt_1mab_t * eps) / (sqrt_ab_t + 1e-12)

        if t == 0:
            x = x0_pred
            break
        t_next = t_subset[idx+1]
        ab_next = sched["alphas_bar"][t_next]
        sqrt_ab_next = torch.sqrt(ab_next)
        sqrt_1mab_next = torch.sqrt(1.0 - ab_next)

        # formula DDIM (Song et al.)
        sigma_t = eta * torch.sqrt((1 - ab_next) / (1 - ab_t) * (1 - ab_t/ab_next))
        dir_xt = (sqrt_1mab_next) * eps
        x = sqrt_ab_next * x0_pred + dir_xt
        if eta > 0 and t_next > 0:
            x = x + sigma_t.view(1,1,1,1) * torch.randn_like(x)

    x = torch.clamp(x, min=0.0)
    s = x.flatten(1).sum(dim=1).view(B,1,1,1) + 1e-12
    return x / s

# -------------------- Plot helpers & metrics --------------------
def imshow_with_individual_colorbar(ax, img2d, title: str, cmap="viridis"):
    im = ax.imshow(img2d, origin="lower", interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(im, cax=cax)

def line_overlay(ax, xs, y_real, y_gen, title: str, xlabel: str):
    ax.plot(xs, y_real, label="Real"); ax.plot(xs, y_gen, linestyle="--", label="Gen")
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel); ax.set_ylabel("counts"); ax.legend(frameon=False)

def normalized_entropy(x: np.ndarray) -> float:
    p = x.astype(np.float64).ravel(); p = np.clip(p, 0.0, None)
    s = p.sum(); 
    if s <= 0: return 0.0
    p = p / s
    Hs = -(p * (np.log(p + EPS))).sum()
    Hmax = math.log(H*W)
    return float(Hs / (Hmax + EPS))

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Inferenza & Valutazione Stage B (DDPM)")
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/", help="Cartella con train_compact/, test_compact/, stats/manifest.json.gz")
    ap.add_argument("--ckpt", default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_B_DDPM/stageB_ddpm_best.pt", help="Checkpoint DDPM (.pt)")
    ap.add_argument("--out-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/infer/stageB_DDPM_Res", help="Dove salvare figure e report")
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--num-sample-plot", type=int, default=8)
    ap.add_argument("--sampler", choices=["ddpm","ddim"], default="ddpm", help="Sampler di inferenza")
    ap.add_argument("--steps", type=int, default=2000, help="Numero di step effettivi (sotto-campionamento della schedule)")
    ap.add_argument("--eta", type=float, default=0.0, help="Rumore DDIM (0 = deterministico)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--recompute-from-image", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ------- Carica ckpt -------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cond_cols = ckpt.get("cond_cols", COND_COLS_DEFAULT)
    T = int(ckpt.get("timesteps", 1000))
    print(f"[INFO] cond_cols dal ckpt: {cond_cols}  |  T={T}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modello (stessa arch del training)
    model = UNetDDPM(cond_dim=len(cond_cols), base=32, tdim=64).to(device)
    # Pesi: se c'è EMA nel ckpt, usala
    if ckpt.get("ema", False) and ("model_ema" in ckpt):
        print("[INFO] Using EMA weights from checkpoint.")
        model.load_state_dict(ckpt["model_ema"], strict=False)
    else:
        model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Schedule dal ckpt (exact betas)
    sched = prepare_schedule_from_ckpt(ckpt, device)

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

    ds_te = ShapeDDPMDataset(rows_test, cond_cols, recompute_from_image=args.recompute_from_image)
    N_test = len(ds_te)
    print(f"[INFO] Test set size: {N_test}")

    # ============================================================
    # 1) CANVA 8x2 — real vs gen con colorbar indipendente
    # ============================================================
    K = min(args.num_sample_plot, N_test)
    idx8 = np.random.choice(N_test, size=K, replace=False)
    reals_8, cond_8 = [], []
    for i in idx8:
        x0, c = ds_te[i]
        reals_8.append(x0.numpy()[0]); cond_8.append(c.numpy())
    cond_8_t = torch.tensor(np.stack(cond_8, axis=0), dtype=torch.float32, device=device)

    with torch.no_grad():
        if args.sampler == "ddpm":
            gen_8_t = ddpm_sample(model, cond_8_t, sched, steps=args.steps, device=device, sigma_scale=1.0)
        else:
            gen_8_t = ddim_sample(model, cond_8_t, sched, steps=args.steps, device=device, eta=args.eta)
    gens_8 = gen_8_t.cpu().numpy()[:,0]

    fig, axes = plt.subplots(K, 2, figsize=(8, 2.4*K), constrained_layout=True)
    if K == 1: axes = np.array([axes])
    for r in range(K):
        imshow_with_individual_colorbar(axes[r,0], reals_8[r], title=f"Real #{idx8[r]}")
        imshow_with_individual_colorbar(axes[r,1], gens_8[r],  title=f"Gen  #{idx8[r]}")
    fig.suptitle("Stage B — DDPM: Real vs Gen (shape centrata)", y=0.995, fontsize=12)
    fig.savefig(out_dir / "grid_real_vs_gen_8x2.png", dpi=150); plt.close(fig)

    # ============================================================
    # 2) Statistiche su campione N
    # ============================================================
    N_eval = min(args.n_eval, N_test)
    idxN = np.random.choice(N_test, size=N_eval, replace=False)
    realsN = np.zeros((N_eval, H, W), dtype=np.float32)
    condN  = np.zeros((N_eval, len(cond_cols)), dtype=np.float32)
    for j, i in enumerate(idxN):
        x0, c = ds_te[i]; realsN[j] = x0.numpy()[0]; condN[j] = c.numpy()

    # Generazione + timing (solo sampling)
    batch = args.batch_size
    genN  = np.zeros_like(realsN)
    t0 = time.time(); 
    if device.type == "cuda": torch.cuda.synchronize()
    with torch.no_grad():
        for s in range(0, N_eval, batch):
            e = min(s+batch, N_eval)
            c_bt = torch.tensor(condN[s:e], dtype=torch.float32, device=device)
            if args.sampler == "ddpm":
                xhat = ddpm_sample(model, c_bt, sched, steps=args.steps, device=device, sigma_scale=1.0)
            else:
                xhat = ddim_sample(model, c_bt, sched, steps=args.steps, device=device, eta=args.eta)
            genN[s:e] = xhat.cpu().numpy()[:,0]
    if device.type == "cuda": torch.cuda.synchronize()
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
    plt.xlabel("Normalized entropy"); plt.ylabel("Density"); plt.title("Entropy — Real vs Gen")
    plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(out_dir / "entropy_hist_real_vs_gen.png", dpi=150); plt.close()

    # ---- Istogrammi 2D: OCCUPANCY (on/off >0) ----
    thr = 0.0 + 1e-12
    occ_real = (realsN > thr).sum(axis=0).astype(np.float64)
    occ_gen  = (genN   > thr).sum(axis=0).astype(np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    imshow_with_individual_colorbar(axes[0], occ_real, "Occupancy (Real)")
    imshow_with_individual_colorbar(axes[1], occ_gen,  "Occupancy (Gen)")
    fig.suptitle(f"Occupancy maps over {N_eval} samples", y=0.98, fontsize=12)
    fig.savefig(out_dir / "occupancy_maps_real_vs_gen.png", dpi=150); plt.close(fig)

    projx_real = occ_real.sum(axis=0); projx_gen = occ_gen.sum(axis=0)
    projy_real = occ_real.sum(axis=1); projy_gen = occ_gen.sum(axis=1)
    xs = np.arange(W); ys = np.arange(H)

    fig, ax = plt.subplots(figsize=(7,3.6))
    line_overlay(ax, xs, projx_real, projx_gen, "Occupancy projection — X", "x (pixel)")
    fig.tight_layout(); fig.savefig(out_dir / "occupancy_proj_x_overlay.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,3.6))
    line_overlay(ax, ys, projy_real, projy_gen, "Occupancy projection — Y", "y (pixel)")
    fig.tight_layout(); fig.savefig(out_dir / "occupancy_proj_y_overlay.png", dpi=150); plt.close(fig)

    # ---- Istogrammi 2D: SOMMA (riempi col valore del pixel) ----
    sum_real = realsN.sum(axis=0).astype(np.float64)
    sum_gen  = genN.sum(axis=0).astype(np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    imshow_with_individual_colorbar(axes[0], sum_real, "Sum map (Real)")
    imshow_with_individual_colorbar(axes[1], sum_gen,  "Sum map (Gen)")
    fig.suptitle(f"Sum maps over {N_eval} samples", y=0.98, fontsize=12)
    fig.savefig(out_dir / "sum_maps_real_vs_gen.png", dpi=150); plt.close(fig)

    projx_real_sum = sum_real.sum(axis=0); projx_gen_sum = sum_gen.sum(axis=0)
    projy_real_sum = sum_real.sum(axis=1); projy_gen_sum = sum_gen.sum(axis=1)

    fig, ax = plt.subplots(figsize=(7,3.6))
    line_overlay(ax, xs, projx_real_sum, projx_gen_sum, "Sum projection — X", "x (pixel)")
    fig.tight_layout(); fig.savefig(out_dir / "sum_proj_x_overlay.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,3.6))
    line_overlay(ax, ys, projy_real_sum, projy_gen_sum, "Sum projection — Y", "y (pixel)")
    fig.tight_layout(); fig.savefig(out_dir / "sum_proj_y_overlay.png", dpi=150); plt.close(fig)

    # ============================================================
    # 3) Curve di training dal CSV + barra su epoca best (robusto)
    # ============================================================
    csv_guess = Path(args.ckpt).parent / "stageB_ddpm_train_log.csv"
    csv_path = csv_guess if csv_guess.exists() else (Path(args.compact_dir) / "stageB_ddpm_train_log.csv")
    if not csv_path.exists():
        print(f"[WARN] Log CSV non trovato in {csv_guess}. Salto il plot delle loss.")
    else:
        df = pd.read_csv(csv_path)
        df["epoch"] = pd.to_numeric(df.get("epoch"), errors="coerce")
        df["val_loss"] = pd.to_numeric(df.get("val_loss"), errors="coerce")
        df["train_loss"] = pd.to_numeric(df.get("train_loss"), errors="coerce")
        df_clean = df.dropna(subset=["epoch"]).reset_index(drop=True)
        if "val_loss" in df_clean.columns and df_clean["val_loss"].notna().any():
            pos = int(np.argmin(df_clean["val_loss"].values))
            best_ep = int(df_clean.loc[pos, "epoch"])
        else:
            best_ep = int(df_clean["epoch"].max())

        plt.figure(figsize=(7.2,4.2))
        if "train_loss" in df.columns: plt.plot(df["epoch"], df["train_loss"], label="Train loss")
        if "val_loss" in df.columns:   plt.plot(df["epoch"], df["val_loss"],  label="Val loss")
        plt.axvline(best_ep, color="k", linestyle=":", linewidth=1.5, label=f"Best epoch = {best_ep}")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Stage B — DDPM Training curves")
        plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(out_dir / "train_val_loss.png", dpi=150); plt.close()

    print(f"[DONE] Output salvati in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

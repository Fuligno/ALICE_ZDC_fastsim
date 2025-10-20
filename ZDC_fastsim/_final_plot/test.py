#!/usr/bin/env python3
# ============================================================
# cfm_helped_quickcheck_integer_shift.py
#
# Quick check: genera 4 matrici FINAL usando
#  - Stage A: impatto (baricentro previsto)
#  - Stage B: CFM vanilla (RAW non centrata, non normalizzata)
#  - Stage C: psum (scala finale)
#
# Ricomposizione GEN:
#   1) bar_B = barycenter_int(gen_raw[i])
#   2) bar_A = round(bar_pred_01[i] * (W-1, H-1))
#   3) scale → round: gen_raw[i] → normalizza a somma=1, scala a psum_gen[i], arrotonda per pixel
#   4) shift intero dx = ax - bxB ; dy = ay - byB (padding a 0, nessun wrap)
#
# Plot: 2 righe × 4 colonne, coppie (Real | Gen) affiancate, scala colori comune per coppia.
# ============================================================

from __future__ import annotations
import argparse, gzip, json, math, random, time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------- Costanti --------------------
H = 44
W = 44
MID_X = (W - 1.0) / 2.0
MID_Y = (H - 1.0) / 2.0
EPS = 1e-12
FEATURE_A = ["E","vx","vy","vz","px","py","pz"]        # Stage A / C

# -------------------- Utils IO --------------------
def load_json_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def find_gamma_col(df: pd.DataFrame) -> str:
    for c in ["Gamma_tot","T","psum","photonSum"]:
        if c in df.columns:
            return c
    raise KeyError("Colonna psum non trovata (alias cercati: Gamma_tot/T/psum/photonSum)")

# -------------------- Stage A: Impact head --------------------
class MLPImpact(nn.Module):
    def __init__(self, in_dim=7, hidden=128, layers=3, dropout=0.0):
        super().__init__()
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
    def forward(self, x):  # -> [0,1]^2
        return self.out_act(self.head(self.backbone(x)))

def load_stageA_and_predict_xy(compact_dir: Path, ckpt_path: Path, device) -> Dict[str, Any]:
    stats = load_json_gz(str(compact_dir / "stats" / "stats.json.gz"))
    mean7 = torch.tensor(stats["mean7"], dtype=torch.float32, device=device)
    std7  = torch.tensor(stats["std7"],  dtype=torch.float32, device=device)
    Hh, Ww = int(stats["H"]), int(stats["W"])

    df = pd.read_parquet(compact_dir / "test_compact" / "test.parquet")
    X = torch.tensor(df[FEATURE_A].to_numpy(np.float32, copy=False), device=device)
    Y01_true = torch.tensor(df[["x_imp","y_imp"]].to_numpy(np.float32, copy=False), device=device)

    Xn = (X - mean7) / (std7 + 1e-6)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["config"]
    model = MLPImpact(in_dim=7, hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    with torch.no_grad():
        Y01_pred = model(Xn).clamp(0.0, 1.0)  # [N,2]

    return dict(
        H=Hh, W=Ww,
        bar_true_01 = Y01_true.cpu().numpy(),   # [N,2]
        bar_pred_01 = Y01_pred.cpu().numpy(),   # [N,2]
        df=df
    )

# -------------------- Stage B: CFM VANILLA --------------------
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

@torch.no_grad()
def make_time_grid(device: torch.device, steps: int, gamma: float):
    u = torch.linspace(0.0, 1.0, steps+1, device=device)
    return torch.pow(1.0 - u, gamma)  # t[0]=1 -> t[-1]=0

@torch.no_grad()
def sample_cfm_vanilla(model: UNetCFM, c: torch.Tensor, steps: int = 200, sigma: float = 1.0,
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
                v1 = model(x, t_hi_t, c); x_pred = x - dt * v1; v2 = model(x_pred, t_lo_t, c); x = x - 0.5*dt*(v1+v2)
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
                v = model(x, t_k, c); x = x - dt * v
            elif scheme == "heun":
                t_km1 = torch.full((B,1), (k-1)*dt, device=device)
                v1 = model(x, t_k, c); x_pred = x - dt*v1
                v2 = model(x_pred, t_km1, c); x = x - 0.5*dt*(v1+v2)
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
    return x  # RAW (scala training)

# -------------------- Stage C: psum predictor (CFM 1D) --------------------
class CFMPredictor(nn.Module):
    def __init__(self, cond_dim=7, hidden=128, layers=3, dropout=0.0):
        super().__init__()
        in_dim = 1 + 3 + cond_dim
        blocks = []
        d = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout > 0: blocks += [nn.Dropout(dropout)]
            d = hidden
        blocks += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*blocks)
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t2 = t*t; t3 = t2*t
        inp = torch.cat([x_t.unsqueeze(1), t.unsqueeze(1), t2.unsqueeze(1), t3.unsqueeze(1), cond], dim=1)
        return self.net(inp).squeeze(1)

@torch.no_grad()
def sample_cfm_scalar(model, cond, steps: int=80):
    device = cond.device
    N = cond.size(0)
    x = torch.randn(N, device=device)
    dt = 1.0/steps
    for s in range(steps):
        t = torch.full((N,), (s + 0.5) * dt, device=device)
        v = model(x, t, cond)
        x = x + v * dt
    return x  # standardized y

# -------------------- INTEGER pipeline: bary, scale&round, shift --------------------
def barycenter_int(img2d: np.ndarray) -> tuple[int, int]:
    g = np.asarray(img2d, dtype=np.float64)
    g = np.clip(g, 0.0, None)
    s = g.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(round((W-1)/2.0)), int(round((H-1)/2.0))
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    by = float((g.sum(axis=1) * ys).sum() / s)
    bx = float((g.sum(axis=0) * xs).sum() / s)
    bx_i = int(np.clip(np.rint(bx), 0, W-1))
    by_i = int(np.clip(np.rint(by), 0, H-1))
    return bx_i, by_i

def scale_by_psum_and_round(img2d: np.ndarray, psum: float) -> np.ndarray:
    g = np.asarray(img2d, dtype=np.float64)
    g = np.clip(g, 0.0, None)
    s = g.sum()
    if s > 0.0 and np.isfinite(s):
        g = g * (float(psum) / s)
    else:
        g.fill(0.0)
    g = np.rint(g)           # arrotonda per pixel
    g[g < 0.0] = 0.0
    return g.astype(np.float32, copy=False)

def shift_integer_2d(img2d: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(img2d, dtype=img2d.dtype)
    # sorgente
    x0_src = max(0, -dx);           x1_src = min(W, W - dx)
    y0_src = max(0, -dy);           y1_src = min(H, H - dy)
    # destinazione
    x0_dst = max(0, dx);            x1_dst = min(W, W + dx)
    y0_dst = max(0, dy);            y1_dst = min(H, H + dy)
    if x0_src < x1_src and y0_src < y1_src:
        out[y0_dst:y1_dst, x0_dst:x1_dst] = img2d[y0_src:y1_src, x0_src:x1_src]
    return out

def compose_final_integer(gen_raw: np.ndarray,
                          bar_gen01: np.ndarray,  # [N,2] in [0,1]
                          psum_gen: np.ndarray) -> np.ndarray:
    N = gen_raw.shape[0]
    out = np.zeros_like(gen_raw, dtype=np.float32)
    for i in range(N):
        # 1) baricentro intero da B (su RAW non normalizzata)
        bxB, byB = barycenter_int(gen_raw[i])

        # 2) punto previsto da A (intero)
        ax = int(np.clip(np.rint(bar_gen01[i,0] * (W-1.0)), 0, W-1))
        ay = int(np.clip(np.rint(bar_gen01[i,1] * (H-1.0)), 0, H-1))

        # 3) scala a psum_gen e arrotonda per pixel
        scaled = scale_by_psum_and_round(gen_raw[i], float(psum_gen[i]))

        # 4) shift intero con padding zero
        dx = ax - bxB
        dy = ay - byB
        out[i] = shift_integer_2d(scaled, dx, dy)
    return out

# -------------------- Plot helpers --------------------
def _root_like_colormap() -> LinearSegmentedColormap:
    colors = [
        (0.00, (0.05, 0.05, 0.35)),
        (0.25, (0.00, 0.45, 0.80)),
        (0.50, (0.05, 0.70, 0.50)),
        (0.75, (0.60, 0.85, 0.20)),
        (1.00, (0.98, 0.92, 0.15)),
    ]
    cmap = LinearSegmentedColormap.from_list("rootlike", colors)
    cmap.set_bad(alpha=0.0)
    return cmap

def _add_cbar_right_of(ax: plt.Axes, mappable, label: str = "", size: str = "3.5%", pad: float = 0.06):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = ax.figure.colorbar(mappable, cax=cax)
    if label:
        cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb

def imshow_pair_grid(final_real: np.ndarray, final_gen: np.ndarray, out_png: Path, dpi: int = 150):
    """
    final_*: [N,H,W], N>=1. Mostra 4 esempi (o meno) in griglia 2x4 con coppie Real|Gen.
    """
    N = final_real.shape[0]
    n_show = min(4, N)
    idx = np.random.choice(N, size=n_show, replace=False)

    fig, axs = plt.subplots(2, 4, figsize=(14.0, 7.0), constrained_layout=False)
    cmap = _root_like_colormap()

    for k, ii in enumerate(idx):
        r = final_real[ii]; g = final_gen[ii]
        row = 0 if k < 2 else 1
        col_pair = (k % 2) * 2
        axL = axs[row, col_pair]
        axR = axs[row, col_pair + 1]

        vmax_pair = float(max(r.max(initial=0.0), g.max(initial=0.0)))
        vmax_pair = max(vmax_pair, 1.0)
        norm_pair = Normalize(vmin=0.0, vmax=vmax_pair)

        imL = axL.imshow(np.ma.masked_less_equal(r, 0.0), origin="lower",
                         interpolation="nearest", cmap=cmap, norm=norm_pair)
        axL.set_title(f"Final Real #{k+1}")
        axL.set_xlabel("x [Pixel]"); axL.set_ylabel("y [Pixel]")
        _add_cbar_right_of(axL, imL, label="Counts")

        axR.imshow(np.ma.masked_less_equal(g, 0.0), origin="lower",
                   interpolation="nearest", cmap=cmap, norm=norm_pair)
        axR.set_title(f"Final Gen  #{k+1}")
        axR.set_xlabel("x [Pixel]"); axR.set_ylabel("y [Pixel]")

        for ax in (axL, axR):
            ax.set_xlim(0, W-1); ax.set_ylim(0, H-1)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="CFM-helped quick check (integer shift compose)")
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/")
    # Stage A
    ap.add_argument("--stageA-ckpt", default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_A/impact_head_best.pt")
    # Stage B (VANILLA)
    ap.add_argument("--stageB-ckpt", default="/data/dataalice/dfuligno/ZDC_fastsim/runs_cfm_vanilla/vanilla_cfm_best.pt")
    ap.add_argument("--stageB-steps", type=int, default=200)
    ap.add_argument("--stageB-sampler", choices=["euler","heun","rk4","heun_nu","rk4_nu"], default="rk4_nu")
    ap.add_argument("--stageB-timegamma", type=float, default=2.5)
    ap.add_argument("--stageB-sigma", type=float, default=1.0)
    # Stage C
    ap.add_argument("--stageC-ckpt", default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_C/gammaCFM_best.pt")
    ap.add_argument("--stageC-steps", type=int, default=80)
    # Quick check
    ap.add_argument("--n", type=int, default=4, help="n esempi da generare (default 4)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-png", default="cfm_helped_quickcheck.png")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compact_dir = Path(args.compact_dir)

    # --------- Stage A: baricentro true/gen ----------
    print("[Stage A] Carico & predico baricentri...")
    A = load_stageA_and_predict_xy(compact_dir, Path(args.stageA_ckpt), device)
    df_te = A["df"]
    N_all = len(df_te)
    bar_true_01 = A["bar_true_01"]     # [N,2]
    bar_pred_01 = A["bar_pred_01"]     # [N,2]

    # --------- Selezione indici ----------
    K = min(int(args.n), N_all)
    idx_sel = np.random.choice(N_all, size=K, replace=False)
    idx_sel.sort()
    print(f"[INFO] N_all={N_all} | K={K}")

    # sottoinsiemi
    X7 = df_te.iloc[idx_sel][FEATURE_A].to_numpy(np.float32, copy=False)
    bar_true_01_K = bar_true_01[idx_sel]
    bar_pred_01_K = bar_pred_01[idx_sel]
    gamma_col = find_gamma_col(df_te)
    psum_true = df_te.iloc[idx_sel][gamma_col].to_numpy(np.float32, copy=False)

    # --------- Stage B: CFM VANILLA → RAW generata ---------
    print("[Stage B] Carico ckpt CFM VANILLA & sampling RAW...")
    ckptB = torch.load(args.stageB_ckpt, map_location="cpu")
    cond_cols_B = ckptB.get("cond_cols", ["E","vx","vy","vz","px","py","pz"])
    base_ch   = int(ckptB.get("base_ch", 32))
    inv_scale = float(ckptB.get("intensity_scale", 1.0))  # in inferenza: dividiamo per questo
    # alcuni checkpoint salvano chiavi con prefissi diversi; gestisci mismatch base/dec2 dimensioni
    modelB = UNetCFM(cond_dim=len(cond_cols_B), base=base_ch).to(device)
    missing, unexpected = modelB.load_state_dict(ckptB["model"], strict=False)
    if unexpected:
        print("[Stage B][WARN] unexpected keys:", unexpected)
    if missing:
        print("[Stage B][WARN] missing keys:", missing)
    modelB.eval()

    condB_np = df_te.iloc[idx_sel][cond_cols_B].to_numpy(np.float32, copy=False)

    gen_raw = np.zeros((K, H, W), dtype=np.float32)
    BATCH = int(args.batch_size)
    n_batches = (K + BATCH - 1) // BATCH

    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for b in range(n_batches):
            s = b * BATCH; e = min(K, s+BATCH)
            c_bt = torch.tensor(condB_np[s:e], dtype=torch.float32, device=device)
            xhat = sample_cfm_vanilla(modelB, c_bt, steps=args.stageB_steps, sigma=args.stageB_sigma,
                                      device=device, scheme=args.stageB_sampler, time_gamma=args.stageB_timegamma)
            if inv_scale != 1.0:
                xhat = xhat / inv_scale
            xhat = torch.clamp(xhat, min=0.0)
            gen_raw[s:e] = xhat[:,0].detach().cpu().numpy()
    if device.type == "cuda": torch.cuda.synchronize()
    print(f"[Stage B] Done. Avg gen time/item ≈ {(time.time()-t0)/float(K):.6f}s")

    # --------- RAW reale (final_real) + shape_real se servisse (qui non usata) ----------
    print("[IO] Carico immagini reali grezze...")
    manifest = load_json_gz(str(compact_dir / "stats" / "manifest.json.gz"))
    test_meta = manifest["test"]
    shard_src = test_meta.get("shard_src", None)
    image_col = test_meta.get("image_col", None)
    lmdb_path = test_meta.get("lmdb", None)

    df_img_cache = None
    env = None
    if (not lmdb_path) and shard_src and image_col:
        df_img_cache = pd.read_pickle(shard_src)
    elif lmdb_path:
        import lmdb
        env = lmdb.open(lmdb_path, readonly=True, lock=False)

    def read_img_by_rowidx(idx: int) -> np.ndarray:
        if env is not None:
            with env.begin() as txn:
                key = f"{idx:08d}".encode("ascii")
                buf = txn.get(key)
                if buf is None:
                    raise KeyError(f"Idx {idx} non trovato in LMDB {lmdb_path}")
                return np.frombuffer(buf, dtype=np.float32).reshape(H, W)
        elif df_img_cache is not None:
            val = df_img_cache.iloc[idx][image_col]
            arr = np.asarray(val)
            if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
            return arr.astype(np.float32, copy=False)
        else:
            raise RuntimeError("Né LMDB né shard_src disponibili per leggere l'immagine reale.")

    final_real = np.zeros((K, H, W), dtype=np.float32)
    for j, ridx in enumerate(idx_sel):
        img = read_img_by_rowidx(int(ridx))
        final_real[j] = np.clip(img, 0.0, None)
    if env is not None:
        env.close()

    # --------- Stage C: psum_gen (float continuo) ----------
    print("[Stage C] Genero psum (CFM 1D)...")
    ckptC = torch.load(args.stageC_ckpt, map_location="cpu")
    cfgC  = ckptC["config"]
    if all(k in cfgC for k in ["mean7","std7","logT_mean","logT_std"]):
        mean7 = np.asarray(cfgC["mean7"], dtype=np.float32)
        std7  = np.asarray(cfgC["std7"],  dtype=np.float32)
        logT_mean = float(cfgC["logT_mean"]); logT_std = float(cfgC["logT_std"])
    else:
        stats = load_json_gz(str(compact_dir / "stats" / "stats.json.gz"))
        mean7 = np.asarray(stats["mean7"], dtype=np.float32)
        std7  = np.asarray(stats["std7"],  dtype=np.float32)
        logT_mean = float(stats["logT_mean"]); logT_std = float(stats["logT_std"])

    X7n = (X7 - mean7[None,:]) / (std7[None,:] + 1e-6)
    modelC = CFMPredictor(cond_dim=7,
                          hidden=cfgC.get("hidden",128),
                          layers=cfgC.get("layers",3),
                          dropout=cfgC.get("dropout",0.0)).to(device)
    modelC.load_state_dict(ckptC["state_dict"]); modelC.eval()

    psum_gen = np.zeros((K,), dtype=np.float32)
    for b in range(n_batches):
        s = b * BATCH; e = min(K, s+BATCH)
        cond_bt = torch.tensor(X7n[s:e], dtype=torch.float32, device=device)
        with torch.no_grad():
            y_std = sample_cfm_scalar(modelC, cond_bt, steps=args.stageC_steps)
            y = y_std * (logT_std + 1e-6) + logT_mean
            T_cont = torch.expm1(y).clamp_min(0.0).float()
        psum_gen[s:e] = T_cont.detach().cpu().numpy()

    # --------- Ricomposizione GEN: integer pipeline ----------
    print("[Compose] scale → round → integer shift ...")
    final_gen = compose_final_integer(
        gen_raw,
        bar_pred_01_K.astype(np.float32),
        psum_gen.astype(np.float32)
    )

    # --------- Plot ----------
    out_png = Path(args.out_png)
    imshow_pair_grid(final_real, final_gen, out_png, dpi=150)
    print(f"[OK] Plot salvato: {out_png.resolve()}")

if __name__ == "__main__":
    main()

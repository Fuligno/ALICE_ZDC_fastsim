#!/usr/bin/env python3
# Inference & plots — Conditional Flow Matching (Stage C)
# - Carica ckpt CFM
# - Genera T_gen per ogni evento del test integrando dx/dt = v_theta(x,t,cond) da t=0→1 (Euler)
# - Spazio modello: y = standardize(log1p(T+U)); in inferenza U non serve: campioniamo x0 e integriamo
# - Denormalizzazione: y -> T_cont = expm1(y*std + mean), clamp≥0, randomized rounding
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

FEATURE_NAMES = ["E","vx","vy","vz","px","py","pz"]
GAMMA_ALIASES = ["Gamma_tot","T","psum","photonSum"]

def find_gamma_col(df: pd.DataFrame) -> str:
    for c in GAMMA_ALIASES:
        if c in df.columns:
            return c
    raise KeyError(f"Nessuna colonna Γ_tot trovata. Alias cercati: {GAMMA_ALIASES}")

def load_stats(compact_dir: Path):
    sp = compact_dir / "stats" / "stats.json.gz"
    with gzip.open(sp, "rt", encoding="utf-8") as f:
        return json.load(f)

# --- modello (stessa definizione del training) ---
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

# --- integrazione deterministica (Euler) del probability-flow ODE ---
@torch.no_grad()
def sample_cfm(model: nn.Module, cond: torch.Tensor, steps: int = 80) -> torch.Tensor:
    """
    cond: (N,7) cond normalize
    Restituisce y_pred_standardized (N,)
    """
    device = cond.device
    N = cond.size(0)
    x = torch.randn(N, device=device)  # x0 ~ N(0,1)
    dt = 1.0 / steps
    for s in range(steps):
        t = torch.full((N,), (s + 0.5) * dt, device=device)  # mid-point
        v = model(x, t, cond)
        x = x + v * dt
    return x  # ≈ y1_standardized

def ecdf(x: np.ndarray):
    x = np.sort(x)
    n = x.size
    y = np.arange(1, n+1) / n
    return x, y

def plot_distributions(T_real: np.ndarray, T_gen: np.ndarray, out_dir: Path, dpi=140):
    T_real = T_real[np.isfinite(T_real) & (T_real >= 0)]
    T_gen  = T_gen[np.isfinite(T_gen)  & (T_gen  >= 0)]
    if T_real.size == 0 or T_gen.size == 0:
        print("[WARN] distribuzioni vuote dopo sanitizzazione: skip.")
        return

    both = np.concatenate([T_real, T_gen])
    qlo = np.quantile(both, 0.001); qhi = np.quantile(both, 0.999)
    if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
        qlo, qhi = float(both.min()), float(both.max())
    span = max(1.0, qhi - qlo)
    lo = max(0.0, qlo - 0.1 * span); hi = qhi + 0.1 * span
    bins = min(200, max(50, int( (hi-lo)/max(1.0, span/100.0) )))

    # PDF + CCDF
    fig, axes = plt.subplots(1, 2, figsize=(12,4.2))
    ax = axes[0]
    ax.hist(T_real, bins=bins, range=(lo,hi), alpha=0.5, label="real", density=True)
    ax.hist(T_gen,  bins=bins, range=(lo,hi), alpha=0.5, label="generated", density=True)
    ax.set_title("Gamma_tot — PDF (density)")
    ax.set_xlabel("counts"); ax.set_ylabel("density")
    ax.grid(True, alpha=0.3); ax.legend()

    xr, yr = ecdf(T_real); xg, yg = ecdf(T_gen)
    ax = axes[1]
    m = (xr >= lo) & (xr <= hi); ax.semilogy(xr[m], 1.0 - yr[m], label="real")
    m = (xg >= lo) & (xg <= hi); ax.semilogy(xg[m], 1.0 - yg[m], label="generated")
    ax.set_title("Gamma_tot — CCDF (log-y)")
    ax.set_xlabel("counts"); ax.set_ylabel("1 - CDF")
    ax.grid(True, which="both", alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "C_distribution_real_vs_generated.png", dpi=dpi)
    plt.close(fig)

    # log1p diagnostic
    Yr = np.log1p(T_real.astype(np.float64))
    Yg = np.log1p(T_gen.astype(np.float64))
    br = np.quantile(Yr, [0.001, 0.999]); bg = np.quantile(Yg, [0.001, 0.999])
    lo_y = max(0.0, min(br[0], bg[0])); hi_y = max(br[1], bg[1])
    bins_y = min(200, max(50, int((hi_y - lo_y)/max(0.02,(hi_y-lo_y)/100.0))))
    fig, ax = plt.subplots(1,1, figsize=(6.5,4.2))
    ax.hist(Yr, bins=bins_y, range=(lo_y,hi_y), alpha=0.5, label="real (log1p)", density=True)
    ax.hist(Yg, bins=bins_y, range=(lo_y,hi_y), alpha=0.5, label="generated (log1p)", density=True)
    ax.set_title("Gamma_tot — log1p space (diagnostic)")
    ax.set_xlabel("log1p(counts)"); ax.set_ylabel("density")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "C_distribution_log1p_diagnostic.png", dpi=dpi)
    plt.close(fig)

def parse_args():
    ap = argparse.ArgumentParser(description="CFM inference & plots (Stage C — Gamma_tot)")
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/",
                    help="Cartella con test_compact/test.parquet e stats/stats.json.gz")
    ap.add_argument("--best-ckpt",   default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_C/gammaCFM_best.pt",
                    help="Checkpoint best del training CFM")
    ap.add_argument("--out-dir",     default="/data/dataalice/dfuligno/ZDC_fastsim/infer/stageC_Res",
                    help="Cartella output plot")
    ap.add_argument("--steps", type=int, default=80, help="Passi Euler per integrazione 0→1")
    ap.add_argument("--dpi", type=int, default=140)
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica ckpt
    ckpt = torch.load(args.best_ckpt, map_location="cpu")
    cfg = ckpt["config"]

    # Stats (dal ckpt se presenti, fallback a stats.json.gz)
    if "mean7" in cfg and "std7" in cfg and "logT_mean" in cfg and "logT_std" in cfg:
        mean7 = np.asarray(cfg["mean7"], dtype=np.float32)
        std7  = np.asarray(cfg["std7"], dtype=np.float32)
        logT_mean = float(cfg["logT_mean"]); logT_std = float(cfg["logT_std"])
    else:
        stats = load_stats(Path(args.compact_dir))
        mean7 = np.asarray(stats["mean7"], dtype=np.float32)
        std7  = np.asarray(stats["std7"], dtype=np.float32)
        logT_mean = float(stats["logT_mean"]); logT_std = float(stats["logT_std"])

    # Test set
    test_pq = Path(args.compact_dir) / "test_compact" / "test.parquet"
    df_te = pd.read_parquet(test_pq)
    gamma_col = find_gamma_col(df_te)
    X_te = df_te[FEATURE_NAMES].to_numpy(np.float32, copy=False)
    T_real = df_te[gamma_col].to_numpy(np.float64, copy=False)

    Xn = (X_te - mean7[None,:]) / (std7[None,:] + 1e-6)
    Xt = torch.from_numpy(Xn).to(device)

    # Modello
    model = CFMPredictor(cond_dim=7,
                         hidden=cfg.get("hidden",128),
                         layers=cfg.get("layers",3),
                         dropout=cfg.get("dropout",0.0)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Sampling
    with torch.no_grad():
        y1_std = sample_cfm(model, Xt, steps=args.steps)  # standardized y
        y = y1_std * (logT_std + 1e-6) + logT_mean
        T_cont = torch.expm1(y).clamp_min(0.0).cpu().numpy()
        # randomized rounding
        U = np.random.rand(T_cont.shape[0]).astype(np.float64)
        T_gen = np.floor(T_cont + U).astype(np.int64)

    # Debug
    print(f"[DEBUG] T_real: min={T_real.min():.1f} max={T_real.max():.1f} mean={T_real.mean():.2f}")
    print(f"[DEBUG] T_gen : min={T_gen.min()} max={T_gen.max()} mean={T_gen.mean():.2f}")

    # Plot distribuzioni
    plot_distributions(T_real.astype(np.int64), T_gen, out_dir, dpi=args.dpi)
    print("[DONE] Plots salvati in:", out_dir.resolve())

if __name__ == "__main__":
    main()

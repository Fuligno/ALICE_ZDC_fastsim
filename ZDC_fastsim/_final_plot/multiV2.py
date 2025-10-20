#!/usr/bin/env python3
# ============================================================
# three_stage_model_plots.py — modelli "3 stage" (CFM 3-stage, DDPM 3-stage)
# ============================================================

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import torch
import math
from typing import Tuple, Optional, List

# ---- immagine 44x44
H = 44
W = 44
MID_X = (W - 1.0) / 2.0  # 21.5
MID_Y = (H - 1.0) / 2.0  # 21.5
EPS = 1e-12

# ---------- Fourier shift batched ----------
def fourier_shift_2d_torch(img: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
    """
    img: [B,1,H,W] float32; tx,ty: [B] in pixel (+x=destra, +y=giu)
    ritorna: [B,1,H,W]
    """
    B, _, Hh, Ww = img.shape
    Fimg = torch.fft.rfft2(img.squeeze(1))                # [B, H, W//2+1]
    yy = torch.fft.fftfreq(Hh, d=1.0, device=img.device)  # [H]
    xx = torch.fft.rfftfreq(Ww, d=1.0, device=img.device) # [W//2+1]
    phase = torch.exp(-2j * math.pi * (
        yy.view(1, Hh, 1) * ty.view(B, 1, 1) +
        xx.view(1, 1, xx.numel()) * tx.view(B, 1, 1)
    ))
    shifted = torch.fft.irfft2(Fimg * phase, s=(Hh, Ww))
    return shifted.unsqueeze(1)

# ---------- helpers grafici ----------
def _fmt_sci(ax, axis="x"):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)

def _fmt_pixel_axis(ax, which="both"):
    # niente scientifica, tick interi
    if which in ("x", "both"):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        fmtx = ScalarFormatter(useMathText=False); fmtx.set_scientific(False)
        ax.xaxis.set_major_formatter(fmtx)
    if which in ("y", "both"):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        fmty = ScalarFormatter(useMathText=False); fmty.set_scientific(False)
        ax.yaxis.set_major_formatter(fmty)

def ecdf_tail_1m(values: np.ndarray):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0]), np.array([1.0])
    x = np.sort(x)
    n = x.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    one_minus_cdf = 1.0 - ranks / n
    mask = one_minus_cdf > 0
    if mask.sum() == 0:
        mask[-1] = True
    return x[mask], one_minus_cdf[mask]

def save_figure(fig: plt.Figure, path: Path, dpi: int, tight: bool = True):
    # Nessun suptitle; layout controllato a chiamata
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- Colormap ROOT-like con zero trasparente ----------
def _root_like_colormap() -> LinearSegmentedColormap:
    # Blu -> ciano -> verde -> giallo
    colors = [
        (0.00, (0.05, 0.05, 0.35)), # deep blue
        (0.25, (0.00, 0.45, 0.80)), # blue-cyan
        (0.50, (0.05, 0.70, 0.50)), # teal-green
        (0.75, (0.60, 0.85, 0.20)), # green-yellow
        (1.00, (0.98, 0.92, 0.15)), # warm yellow
    ]
    cmap = LinearSegmentedColormap.from_list("rootlike", colors)
    cmap.set_bad(alpha=0.0)  # zero trasparente (via maschera)
    return cmap

def _add_cbar_right_of(ax: plt.Axes, mappable, label: str = "", size: str = "3.5%", pad: float = 0.06):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = ax.figure.colorbar(mappable, cax=cax)
    if label:
        cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb

# ---------- primitives 2D con zeri trasparenti (norm comune a coppia) ----------
from matplotlib.colors import Normalize

def _hist2d_mappable(ax, x, y, bins=(80, 80), xrange=None, yrange=None,
                     title: str = "", xlabel: str = "", ylabel: str = "",
                     pixel_axes: bool = False, mid_lines: Optional[Tuple[float,float]] = None,
                     cmap=None, norm: Optional[Normalize] = None):
    Hh, xe, ye = np.histogram2d(x, y, bins=bins, range=[xrange, yrange])
    data = np.ma.masked_less_equal(Hh.T, 0.0)  # maschera zeri → trasparenti
    im = ax.pcolormesh(xe, ye, data, shading="flat", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=11, pad=6)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    (_fmt_pixel_axis if pixel_axes else _fmt_sci)(ax, "both")
    ax.tick_params(axis="both", labelsize=9)
    if mid_lines is not None:
        mx, my = mid_lines
        ax.axvline(mx, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(my, color="red", linestyle="--", linewidth=1.0)
    return im, Hh

def _imshow_mappable(ax, img2d, title: str, xlabel: str, ylabel: str,
                     pixel_axes: bool, mid_lines: Optional[Tuple[float,float]] = None,
                     cmap=None, norm: Optional[Normalize] = None):
    data = np.ma.masked_less_equal(img2d, 0.0)
    im = ax.imshow(data, origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    (_fmt_pixel_axis if pixel_axes else _fmt_sci)(ax, "both")
    ax.tick_params(axis="both", labelsize=9)
    if mid_lines is not None:
        mx, my = mid_lines
        ax.axvline(mx, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(my, color="red", linestyle="--", linewidth=1.0)
    return im

# ---------- 1D overlay + ratio attaccato ----------
def hist1d_overlay(ax, real, gen, bins_edges, xlabel: str, title: str,
                   pixel_axis: bool = False, mid_line: float | None = None):
    ax.hist(real, bins=bins_edges, density=True, alpha=0.6, label="Real")
    ax.hist(gen,  bins=bins_edges, density=True, alpha=0.6, histtype="step", linewidth=1.8, label="Gen")
    ax.set_xlabel("")  # niente label sopra (sta nel ratio sotto)
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=11, pad=6)
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.2, linestyle=":")
    (_fmt_pixel_axis if pixel_axis else _fmt_sci)(ax, "x")
    if mid_line is not None:
        ax.axvline(mid_line, color="black", linestyle="--", linewidth=1.0)
    ax.tick_params(axis="both", labelsize=9)
    ax.tick_params(axis="x", labelbottom=False)  # toglie le label x sopra

def overlay_plus_ratio(col_ax_top: plt.Axes, col_ax_bottom: plt.Axes,
                       x_values, y_real, y_gen,
                       title_top: str, xlabel_bottom: str, pixel_axis: bool = False,
                       mid_line: float | None = None):
    # top
    col_ax_top.plot(x_values, y_real, label="Real")
    col_ax_top.plot(x_values, y_gen, linestyle="--", label="Gen")
    col_ax_top.set_title(title_top, fontsize=11, pad=6)
    col_ax_top.set_xlabel("")
    col_ax_top.set_ylabel("counts")
    col_ax_top.legend(frameon=False, fontsize=9); col_ax_top.grid(alpha=0.2, linestyle=":")
    (_fmt_pixel_axis if pixel_axis else _fmt_sci)(col_ax_top, "x")
    if mid_line is not None:
        col_ax_top.axvline(mid_line, color="black", linestyle="--", linewidth=1.0)
    col_ax_top.tick_params(axis="both", labelsize=9)
    col_ax_top.tick_params(axis="x", labelbottom=False)

    # bottom: ratio (0..2)
    denom = np.clip(y_real, EPS, None)
    ratio = y_gen / denom
    col_ax_bottom.plot(x_values, ratio, linewidth=1.6)
    col_ax_bottom.axhline(1.0, linestyle="--", linewidth=1.0, color="black")
    col_ax_bottom.set_ylim(0.0, 2.0)
    col_ax_bottom.set_ylabel("Gen / Real")
    col_ax_bottom.set_xlabel(xlabel_bottom)
    col_ax_bottom.grid(alpha=0.2, linestyle=":")
    (_fmt_pixel_axis if pixel_axis else _fmt_sci)(col_ax_bottom, "x")
    col_ax_bottom.tick_params(axis="both", labelsize=9)

# ---------- util: lettura colonne immagine ----------
def _stack_images(col) -> np.ndarray:
    mats = [np.asarray(v, dtype=np.float32) for v in col]
    arr = np.vstack(mats)
    return arr.reshape(-1, H, W)

# ---------- centratura ----------
def center_images_fft(imgs: np.ndarray, bx01: np.ndarray, by01: np.ndarray,
                      device: torch.device | None = None) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(imgs).to(device=device, dtype=torch.float32).unsqueeze(1)  # [N,1,H,W]
    cx = torch.from_numpy((bx01 * (W - 1.0)).astype(np.float32)).to(device)
    cy = torch.from_numpy((by01 * (H - 1.0)).astype(np.float32)).to(device)
    tx = torch.full_like(cx, float(MID_X)) - cx
    ty = torch.full_like(cy, float(MID_Y)) - cy
    with torch.no_grad():
        x_shift = fourier_shift_2d_torch(x, tx, ty).clamp_min(0.0)
    return x_shift.squeeze(1).cpu().numpy()

# ---------- Metriche ----------
def _wd_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=np.float64); b = b[np.isfinite(b)]
    if a.size == 0 and b.size == 0: return 0.0
    if a.size == 0: return float(np.mean(np.abs(b)))
    if b.size == 0: return float(np.mean(np.abs(a)))
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        qs = np.linspace(0.0, 1.0, 1025)
        qa = np.quantile(a, qs, method="linear")
        qb = np.quantile(b, qs, method="linear")
        return float(np.mean(np.abs(qa - qb)))

def _sliced_wd_2d(P: np.ndarray, Q: np.ndarray, n_proj: int = 128, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    if P.size == 0 and Q.size == 0: return 0.0
    thetas = rng.normal(size=(n_proj, 2))
    thetas /= np.linalg.norm(thetas, axis=1, keepdims=True) + 1e-12
    vals = []
    for u in thetas:
        vals.append(_wd_1d(P @ u, Q @ u))
    return float(np.mean(vals))

def _sample_points_from_weighted_grid(grid: np.ndarray, max_samples: int = 30000, seed: int = 42) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float64)
    g[g < 0] = 0.0
    total = g.sum()
    if total <= 0:
        return np.empty((0,2), dtype=np.float64)
    probs = (g / total).ravel()
    n = min(int(max_samples), probs.size)
    rng = np.random.default_rng(seed)
    idx = rng.choice(probs.size, size=n, replace=True, p=probs)
    ys, xs = np.divmod(idx, g.shape[1])
    return np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

def _entropy_per_image(imgs: np.ndarray) -> Tuple[float, float]:
    ent = []
    for m in imgs:
        s = float(m.sum())
        if s <= 0:
            ent.append(0.0)
        else:
            p = m.ravel() / s
            p = p[p > 0]
            ent.append(float(-np.sum(p * np.log(p))))
    ent = np.asarray(ent, dtype=np.float64)
    return float(ent.mean() if ent.size else 0.0), float(ent.var(ddof=0) if ent.size else 0.0)

def _sliced_wd_images(reals: np.ndarray, gens: np.ndarray, n_proj: int = 256, max_n: int = 4000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    A = reals.reshape(reals.shape[0], -1).astype(np.float32, copy=False)
    B = gens.reshape(gens.shape[0], -1).astype(np.float32, copy=False)
    n = int(min(max_n, A.shape[0], B.shape[0]))
    if A.shape[0] > n:
        A = A[rng.choice(A.shape[0], size=n, replace=False)]
    if B.shape[0] > n:
        B = B[rng.choice(B.shape[0], size=n, replace=False)]
    if n == 0: return 0.0
    D = A.shape[1]
    dirs = rng.normal(size=(n_proj, D)).astype(np.float32)
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    PA = A @ dirs.T
    PB = B @ dirs.T
    vals = []
    for j in range(PA.shape[1]):
        a = np.sort(PA[:, j]); b = np.sort(PB[:, j])
        vals.append(float(np.mean(np.abs(a - b))))
    return float(np.mean(vals) if vals else 0.0)
def dual_hist_and_tail(ax_hist, ax_tail, real, gen, title_left: str, xlab: str, bins: int = 60):
    real = np.clip(np.asarray(real, dtype=np.float64), 0.0, None)
    gen  = np.clip(np.asarray(gen,  dtype=np.float64), 0.0, None)

    xmax = float(max(real.max(initial=0.0), gen.max(initial=0.0)))
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0
    bins_edges = np.linspace(0.0, xmax, bins + 1)

    # pannello sinistro: istogrammi sovrapposti
    ax_hist.hist(real, bins=bins_edges, density=True, alpha=0.6, label="Real")
    ax_hist.hist(gen,  bins=bins_edges, density=True, alpha=0.6, histtype="step", linewidth=1.8, label="Gen")
    ax_hist.set_xlabel(xlab)
    ax_hist.set_ylabel("density")
    ax_hist.set_title(title_left, fontsize=11, pad=6)
    ax_hist.legend(frameon=False, fontsize=9)
    ax_hist.grid(alpha=0.2, linestyle=":")
    _fmt_sci(ax_hist, "x")
    ax_hist.tick_params(axis="both", labelsize=9)

    # pannello destro: tail (1-CDF) in log-y
    xr, yr = ecdf_tail_1m(real)
    xg, yg = ecdf_tail_1m(gen)
    ax_tail.plot(xr, yr, label="Real")
    ax_tail.plot(xg, yg, linestyle="--", label="Gen")
    ax_tail.set_yscale("log")
    ax_tail.set_xlabel(xlab)
    ax_tail.set_ylabel("1-CDF")
    ax_tail.set_title("Tail (1−CDF)", fontsize=11, pad=6)
    ax_tail.legend(frameon=False, fontsize=9)
    ax_tail.grid(alpha=0.2, which="both", linestyle=":")
    _fmt_sci(ax_tail, "x")
    ax_tail.tick_params(axis="both", labelsize=9)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Plot macro per modelli 3-stage (CFM/DDPM 3-stage)")
    ap.add_argument("--in", dest="in_path", default="/data/dataalice/dfuligno/ZDC_fastsim/build_data/CFM_helped.parquet",
                    help="Parquet 3-stage (contiene bar_x/y_*, psum_*, pmax_*, shape_*, final_*)")
    ap.add_argument("--out-dir", default="CFM_helped",
                    help="Cartella in cui salvare le figure")
    ap.add_argument("--bins", type=int, default=60, help="Numero di bin per gli istogrammi 1D")
    ap.add_argument("--bins2d", type=int, default=80, help="Numero di bin per istogrammi 2D (per dimensione)")
    ap.add_argument("--dpi", type=int, default=150, help="DPI per i PNG")
    ap.add_argument("--bins-bary-x", type=int, default=W, help="Bin istogramma 1D barycenter X [Pixel]")
    ap.add_argument("--bins-bary-y", type=int, default=H, help="Bin istogramma 1D barycenter Y [Pixel]")
    ap.add_argument("--swd-proj", type=int, default=128, help="N. proiezioni per Sliced WD 2D")
    ap.add_argument("--swd-img-proj", type=int, default=256, help="N. proiezioni per Sliced WD su immagini intere (alta dimensione)")
    ap.add_argument("--swd-img-maxn", type=int, default=4000, help="Max immagini per lato per Sliced WD immagini (sottocampionamento)")
    ap.add_argument("--metrics-on-shape", action="store_true",
                    help="Calcola SWD_images_HD ed entropie anche per le immagini SHAPE (oltre alle FINAL).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.in_path)
    needed = [
        "psum_true","psum_gen","pmax_true","pmax_gen",
        "bar_x_true","bar_y_true","bar_x_gen","bar_y_gen",
        "shape_real","shape_gen","final_real","final_gen"
    ]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Colonna mancante: {c}")

    # --- scalari
    psum_true = df["psum_true"].to_numpy(np.float64, copy=False)
    psum_gen  = df["psum_gen"].to_numpy(np.float64, copy=False)
    pmax_true = df["pmax_true"].to_numpy(np.float64, copy=False)
    pmax_gen  = df["pmax_gen"].to_numpy(np.float64, copy=False)

    # -------- Canvas 1: photon sum --------
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=False)
    dual_hist_and_tail(axs[0], axs[1], psum_true, psum_gen,
                       title_left="Histogram overlap",
                       xlab="Counts", bins=args.bins)
    save_figure(fig, out_dir / "psum_comparison.png", args.dpi)

    # -------- Canvas 2: pmax --------
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=False)
    dual_hist_and_tail(axs[0], axs[1], pmax_true, pmax_gen,
                       title_left="Histogram overlap",
                       xlab="Counts", bins=args.bins)
    save_figure(fig, out_dir / "pmax_comparison.png", args.dpi)

    # -------- Canvas 3A: 2D hist (psum vs pmax) con scala comune --------
    x_max = float(max(np.nanmax(psum_true), np.nanmax(psum_gen))); x_max = 1.0 if not np.isfinite(x_max) else max(1.0, x_max)
    y_max = float(max(np.nanmax(pmax_true), np.nanmax(pmax_gen))); y_max = 1.0 if not np.isfinite(y_max) else max(1.0, y_max)
    xrange = (0.0, x_max); yrange = (0.0, y_max)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=False)
    cmap = _root_like_colormap()
    H1, _, _ = np.histogram2d(psum_true, pmax_true, bins=(args.bins2d, args.bins2d), range=[xrange, yrange])
    H2, _, _ = np.histogram2d(psum_gen,  pmax_gen,  bins=(args.bins2d, args.bins2d), range=[xrange, yrange])
    vmax = float(max(H1.max(initial=0.0), H2.max(initial=0.0)))
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1.0))
    imL, _ = _hist2d_mappable(axs[0], psum_true, pmax_true, bins=(args.bins2d, args.bins2d),
                              xrange=xrange, yrange=yrange, title="Real: 2D histogram",
                              xlabel="Photon sum", ylabel="Peak value",
                              pixel_axes=False, mid_lines=None, cmap=cmap, norm=norm)
    imR, _ = _hist2d_mappable(axs[1], psum_gen,  pmax_gen,  bins=(args.bins2d, args.bins2d),
                              xrange=xrange, yrange=yrange, title="Generated: 2D histogram",
                              xlabel="Photon sum", ylabel="Peak value",
                              pixel_axes=False, mid_lines=None, cmap=cmap, norm=norm)
    _add_cbar_right_of(axs[0], imL, label="Counts")
    save_figure(fig, out_dir / "psum_pmax_2D.png", args.dpi)

    # -------- Canvas 4A: Impact point — solo 2D --------
    bx_t_px = (df["bar_x_true"].to_numpy(np.float64, copy=False)) * (W - 1.0)
    by_t_px = (df["bar_y_true"].to_numpy(np.float64, copy=False)) * (H - 1.0)
    bx_g_px = (df["bar_x_gen"].to_numpy(np.float64,  copy=False)) * (W - 1.0)
    by_g_px = (df["bar_y_gen"].to_numpy(np.float64,  copy=False)) * (H - 1.0)

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=False)
    xy_range = ([0.0, W - 1.0], [0.0, H - 1.0]); bins2d_bary = (args.bins2d, args.bins2d)
    Hb1, _, _ = np.histogram2d(bx_t_px, by_t_px, bins=bins2d_bary, range=[xy_range[0], xy_range[1]])
    Hb2, _, _ = np.histogram2d(bx_g_px, by_g_px, bins=bins2d_bary, range=[xy_range[0], xy_range[1]])
    vmax_b = float(max(Hb1.max(initial=0.0), Hb2.max(initial=0.0)))
    norm_b = Normalize(vmin=0.0, vmax=max(vmax_b, 1.0))
    imL, _ = _hist2d_mappable(axs[0], bx_t_px, by_t_px, bins=bins2d_bary,
                              xrange=xy_range[0], yrange=xy_range[1],
                              title="Real: impact 2D histogram",
                              xlabel="Barycenter x [Pixel]", ylabel="Barycenter y [Pixel]",
                              pixel_axes=True, mid_lines=(MID_X, MID_Y),
                              cmap=cmap, norm=norm_b)
    imR, _ = _hist2d_mappable(axs[1], bx_g_px, by_g_px, bins=bins2d_bary,
                              xrange=xy_range[0], yrange=xy_range[1],
                              title="Generated: impact 2D histogram",
                              xlabel="Barycenter x [Pixel]", ylabel="Barycenter y [Pixel]",
                              pixel_axes=True, mid_lines=(MID_X, MID_Y),
                              cmap=cmap, norm=norm_b)
    _add_cbar_right_of(axs[0], imL, label="Counts")
    save_figure(fig, out_dir / "impact_point_2D_only.png", args.dpi)

    # -------- Canvas 4B: Impact — istogrammi X/Y + ratio --------
    x_edges = np.linspace(0.0, W - 1.0, int(args.bins_bary_x) + 1)
    y_edges = np.linspace(0.0, H - 1.0, int(args.bins_bary_y) + 1)
    x_centers = 0.5*(x_edges[1:] + x_edges[:-1])
    y_centers = 0.5*(y_edges[1:] + y_edges[:-1])
    hx_r, _ = np.histogram(bx_t_px, bins=x_edges, density=True)
    hx_g, _ = np.histogram(bx_g_px, bins=x_edges, density=True)
    hy_r, _ = np.histogram(by_t_px, bins=y_edges, density=True)
    hy_g, _ = np.histogram(by_g_px, bins=y_edges, density=True)

    fig = plt.figure(figsize=(12.2, 7.6))
    gs = fig.add_gridspec(ncols=2, nrows=2, height_ratios=[3,1], wspace=0.28, hspace=0.0)
    ax_x_top   = fig.add_subplot(gs[0,0]); ax_x_ratio = fig.add_subplot(gs[1,0], sharex=ax_x_top)
    ax_y_top   = fig.add_subplot(gs[0,1]); ax_y_ratio = fig.add_subplot(gs[1,1], sharex=ax_y_top)

    overlay_plus_ratio(ax_x_top, ax_x_ratio, x_centers, hx_r, hx_g,
                       title_top="Histogram overlap — x", xlabel_bottom="barycenter x [Pixel]",
                       pixel_axis=True, mid_line=MID_X)
    overlay_plus_ratio(ax_y_top, ax_y_ratio, y_centers, hy_r, hy_g,
                       title_top="Histogram overlap — y", xlabel_bottom="barycenter y [Pixel]",
                       pixel_axis=True, mid_line=MID_Y)
    save_figure(fig, out_dir / "impact_point_profiles_ratio.png", args.dpi, tight=False)

    # ============ Canvas 5A: OCCUPANCY (centered) — FINAL solo 2D ============
    finals_real = _stack_images(df["final_real"])
    finals_gen  = _stack_images(df["final_gen"])

    bx_t01 = df["bar_x_true"].to_numpy(np.float32, copy=False)
    by_t01 = df["bar_y_true"].to_numpy(np.float32, copy=False)
    bx_g01 = df["bar_x_gen"].to_numpy(np.float32,  copy=False)
    by_g01 = df["bar_y_gen"].to_numpy(np.float32,  copy=False)

    finals_real_c = center_images_fft(finals_real, bx_t01, by_t01)
    finals_gen_c  = center_images_fft(finals_gen,  bx_g01, by_g01)

    occ_real = (finals_real_c >= 1.0).sum(axis=0).astype(np.float64)
    occ_gen  = (finals_gen_c  >= 1.0).sum(axis=0).astype(np.float64)

    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=False)
    vmax_occ = float(max(occ_real.max(initial=0.0), occ_gen.max(initial=0.0)))
    norm_occ = Normalize(vmin=0.0, vmax=max(vmax_occ, 1.0))
    imL = _imshow_mappable(axs[0], occ_real, "Real — occupancy (centered, final)",
                           "x [Pixel]", "y [Pixel]", pixel_axes=True,
                           mid_lines=(MID_X, MID_Y), cmap=cmap, norm=norm_occ)
    imR = _imshow_mappable(axs[1], occ_gen,  "Generated — occupancy (centered, final)",
                           "x [Pixel]", "y [Pixel]", pixel_axes=True,
                           mid_lines=(MID_X, MID_Y), cmap=cmap, norm=norm_occ)
    _add_cbar_right_of(axs[0], imL, label="Counts")
    save_figure(fig, out_dir / "centered_occupancy_final_2D_only.png", args.dpi)

    # ============ Canvas 5B: OCCUPANCY — proiezioni + ratio ============
    projx_real = occ_real.sum(axis=0); projx_gen = occ_gen.sum(axis=0)
    projy_real = occ_real.sum(axis=1); projy_gen = occ_gen.sum(axis=1)
    xs = np.arange(W); ys = np.arange(H)

    fig = plt.figure(figsize=(12.2, 7.6))
    gs = fig.add_gridspec(ncols=2, nrows=2, height_ratios=[3,1], wspace=0.28, hspace=0.0)
    ax_xtop   = fig.add_subplot(gs[0,0]); ax_xratio = fig.add_subplot(gs[1,0], sharex=ax_xtop)
    ax_ytop   = fig.add_subplot(gs[0,1]); ax_yratio = fig.add_subplot(gs[1,1], sharex=ax_ytop)

    overlay_plus_ratio(ax_xtop, ax_xratio, xs, projx_real, projx_gen,
                       title_top="X projection (occupancy)", xlabel_bottom="x [Pixel]",
                       pixel_axis=True, mid_line=MID_X)
    overlay_plus_ratio(ax_ytop, ax_yratio, ys, projy_real, projy_gen,
                       title_top="Y projection (occupancy)", xlabel_bottom="y [Pixel]",
                       pixel_axis=True, mid_line=MID_Y)
    save_figure(fig, out_dir / "centered_occupancy_final_profiles_ratio.png", args.dpi, tight=False)

    # ============ Canvas 6A: SUM (centered) — FINAL solo 2D ============
    sum_real = finals_real_c.sum(axis=0).astype(np.float64)
    sum_gen  = finals_gen_c.sum(axis=0).astype(np.float64)

    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=False)
    vmax_sum = float(max(sum_real.max(initial=0.0), sum_gen.max(initial=0.0)))
    norm_sum = Normalize(vmin=0.0, vmax=max(vmax_sum, 1.0))
    imL = _imshow_mappable(axs[0], sum_real, "Real — sum (centered, final)",
                           "x [Pixel]", "y [Pixel]", pixel_axes=True,
                           mid_lines=(MID_X, MID_Y), cmap=cmap, norm=norm_sum)
    imR = _imshow_mappable(axs[1], sum_gen,  "Generated — sum (centered, final)",
                           "x [Pixel]", "y [Pixel]", pixel_axes=True,
                           mid_lines=(MID_X, MID_Y), cmap=cmap, norm=norm_sum)
    _add_cbar_right_of(axs[0], imL, label="sum")
    save_figure(fig, out_dir / "centered_sum_final_2D_only.png", args.dpi)

    # ============ Canvas 6B: SUM — proiezioni + ratio ============
    projx_real_s = sum_real.sum(axis=0); projx_gen_s = sum_gen.sum(axis=0)
    projy_real_s = sum_real.sum(axis=1); projy_gen_s = sum_gen.sum(axis=1)

    fig = plt.figure(figsize=(12.2, 7.6))
    gs = fig.add_gridspec(ncols=2, nrows=2, height_ratios=[3,1], wspace=0.28, hspace=0.0)
    ax_xtop   = fig.add_subplot(gs[0,0]); ax_xratio = fig.add_subplot(gs[1,0], sharex=ax_xtop)
    ax_ytop   = fig.add_subplot(gs[0,1]); ax_yratio = fig.add_subplot(gs[1,1], sharex=ax_ytop)

    overlay_plus_ratio(ax_xtop, ax_xratio, xs, projx_real_s, projx_gen_s,
                       title_top="X projection (sum)", xlabel_bottom="x [Pixel]",
                       pixel_axis=True, mid_line=MID_X)
    overlay_plus_ratio(ax_ytop, ax_yratio, ys, projy_real_s, projy_gen_s,
                       title_top="Y projection (sum)", xlabel_bottom="y [Pixel]",
                       pixel_axis=True, mid_line=MID_Y)
    save_figure(fig, out_dir / "centered_sum_final_profiles_ratio.png", args.dpi, tight=False)

    # ============ Canvas 7: Random samples (FINAL) — coppie con scala comune ============
    n_total_f = finals_real.shape[0]
    n_show_f = min(4, n_total_f)
    sel_f = np.random.choice(n_total_f, size=n_show_f, replace=False)

    fig, axs = plt.subplots(2, 4, figsize=(12.8, 6.8), constrained_layout=False)
    for k, idx in enumerate(sel_f):
        r = finals_real[idx]; g = finals_gen[idx]
        row = 0 if k < 2 else 1
        col_pair = (k % 2) * 2
        ax_l = axs[row, col_pair]; ax_r = axs[row, col_pair + 1]

        vmax_pair = float(max(r.max(initial=0.0), g.max(initial=0.0)))
        norm_pair = Normalize(vmin=0.0, vmax=max(vmax_pair, 1.0))

        im_l = _imshow_mappable(ax_l, r, f"Final Real #{k+1}",
                                "x [Pixel]", "y [Pixel]", pixel_axes=True, cmap=cmap, norm=norm_pair)
        im_r = _imshow_mappable(ax_r, g, f"Final Gen #{k+1}",
                                "x [Pixel]", "y [Pixel]", pixel_axes=True, cmap=cmap, norm=norm_pair)
        _add_cbar_right_of(ax_l, im_l, label="Pixel counts")
    save_figure(fig, out_dir / "samples_final_real_vs_gen.png", args.dpi)

    # ============ Canvas 8: Random samples (SHAPE) — coppie con scala comune ============
    shapes_real = _stack_images(df["shape_real"])
    shapes_gen  = _stack_images(df["shape_gen"])

    n_total_s = shapes_real.shape[0]
    n_show_s = min(4, n_total_s)
    sel_s = np.random.choice(n_total_s, size=n_show_s, replace=False)

    fig, axs = plt.subplots(2, 4, figsize=(12.8, 6.8), constrained_layout=False)
    for k, idx in enumerate(sel_s):
        r = shapes_real[idx]; g = shapes_gen[idx]
        row = 0 if k < 2 else 1
        col_pair = (k % 2) * 2
        ax_l = axs[row, col_pair]; ax_r = axs[row, col_pair + 1]

        vmax_pair = float(max(r.max(initial=0.0), g.max(initial=0.0)))
        norm_pair = Normalize(vmin=0.0, vmax=max(vmax_pair, 1.0))

        im_l = _imshow_mappable(ax_l, r, f"Shape Real #{k+1} (idx {int(idx)})",
                                "x [Pixel]", "y [Pixel]", pixel_axes=True, cmap=cmap, norm=norm_pair)
        im_r = _imshow_mappable(ax_r, g, f"Shape Gen #{k+1} (idx {int(idx)})",
                                "x [Pixel]", "y [Pixel]", pixel_axes=True, cmap=cmap, norm=norm_pair)
        _add_cbar_right_of(ax_l, im_l, label="Pixel counts")
    save_figure(fig, out_dir / "samples_shape_real_vs_gen.png", args.dpi)

    # ==================== METRICHE: WD + Entropie ====================
    metrics_lines: List[str] = []

    # 1D WD su psum e pmax
    wd_psum = _wd_1d(psum_true, psum_gen)
    wd_pmax = _wd_1d(pmax_true, pmax_gen)
    metrics_lines.append(f"WD_1D psum: {wd_psum:.6g}")
    metrics_lines.append(f"WD_1D pmax: {wd_pmax:.6g}")

    # 2D Sliced WD: (psum,pmax)
    P_pp = np.column_stack([psum_true, pmax_true])
    Q_pp = np.column_stack([psum_gen,  pmax_gen])
    swd_pp = _sliced_wd_2d(P_pp, Q_pp, n_proj=int(args.swd_proj), seed=123)
    metrics_lines.append(f"SlicedWD_2D (psum,pmax): {swd_pp:.6g}")

    # 2D Sliced WD: Impact (bary x,y) in pixel
    P_imp = np.column_stack([bx_t_px, by_t_px])
    Q_imp = np.column_stack([bx_g_px, by_g_px])
    swd_imp = _sliced_wd_2d(P_imp, Q_imp, n_proj=int(args.swd_proj), seed=321)
    metrics_lines.append(f"SlicedWD_2D impact (x,y): {swd_imp:.6g}")

    # 2D Sliced WD: Occupancy maps (campionamento pesato) — FINAL
    pts_occ_real = _sample_points_from_weighted_grid(occ_real, max_samples=30000, seed=1)
    pts_occ_gen  = _sample_points_from_weighted_grid(occ_gen,  max_samples=30000, seed=2)
    swd_occ = _sliced_wd_2d(pts_occ_real, pts_occ_gen, n_proj=int(args.swd_proj), seed=777)
    metrics_lines.append(f"SlicedWD_2D occupancy_final: {swd_occ:.6g}")

    # 2D Sliced WD: Sum maps (campionamento pesato) — FINAL
    pts_sum_real = _sample_points_from_weighted_grid(sum_real, max_samples=30000, seed=3)
    pts_sum_gen  = _sample_points_from_weighted_grid(sum_gen,  max_samples=30000, seed=4)
    swd_sum = _sliced_wd_2d(pts_sum_real, pts_sum_gen, n_proj=int(args.swd_proj), seed=888)
    metrics_lines.append(f"SlicedWD_2D sum_final: {swd_sum:.6g}")

    # Sliced WD in alta dimensione su immagini intere — FINAL (obbligatorio)
    swd_imgs_final = _sliced_wd_images(finals_real, finals_gen,
                                       n_proj=int(args.swd_img_proj),
                                       max_n=int(args.swd_img_maxn), seed=42)
    metrics_lines.append(f"SlicedWD_images_HD_final: {swd_imgs_final:.6g}")

    # (Opzionale) Sliced WD immagini — SHAPE
    if args.metrics_on_shape:
        swd_imgs_shape = _sliced_wd_images(shapes_real, shapes_gen,
                                           n_proj=int(args.swd_img_proj),
                                           max_n=int(args.swd_img_maxn), seed=43)
        metrics_lines.append(f"SlicedWD_images_HD_shape: {swd_imgs_shape:.6g}")

    # Entropie (media,var) per-matrice — FINAL
    ent_real_mean, ent_real_var = _entropy_per_image(finals_real)
    ent_gen_mean,  ent_gen_var  = _entropy_per_image(finals_gen)
    metrics_lines.append(f"Entropy final real: mean={ent_real_mean:.6g}, var={ent_real_var:.6g}")
    metrics_lines.append(f"Entropy final gen:  mean={ent_gen_mean:.6g}, var={ent_gen_var:.6g}")

    # Scrivi su file
    with open(out_dir / "metrics.txt", "w") as f:
        f.write("\n".join(metrics_lines) + "\n")

    print(f"[OK] Salvate figure in: {out_dir.resolve()}")
    print("[METRICS]\n" + "\n".join(metrics_lines))

if __name__ == "__main__":
    main()

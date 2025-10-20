#!/usr/bin/env python3
# ============================================================
# three_stage_model_plots.py — modelli "3 stage" (CFM 3-stage, DDPM 3-stage)
# Canvas:
# 1) psum: hist overlap + 1-CDF (log)
# 2) pmax: hist overlap + 1-CDF (log)
# 3) 2D hist (psum vs pmax) — real vs gen
# 4) Impact point (2x2): 2D hist pixel + overlay 1D (x,y)    [usa bar_x/bar_y in [0,1] → pixel]
# 5) Occupancy (centered) — real(final) vs gen(final) + profili X/Y
# 6) Sum (centered) — real(final) vs gen(final) + profili X/Y
# 7) Random samples (final): 4 coppie Real/Gen
# 8) Random samples (shape): 4 coppie Real/Gen
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
import torch
import math

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

def save_figure(fig: plt.Figure, path: Path, suptitle: str, dpi: int):
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.subplots_adjust(top=0.88)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- primitives di plot ----------
def dual_hist_and_tail(ax_hist, ax_tail, real, gen, title_left: str, xlab: str, bins: int = 60):
    real = np.clip(np.asarray(real, dtype=np.float64), 0.0, None)
    gen  = np.clip(np.asarray(gen,  dtype=np.float64), 0.0, None)
    xmax = float(max(real.max(initial=0.0), gen.max(initial=0.0)))
    if not np.isfinite(xmax) or xmax <= 0: xmax = 1.0
    bins_edges = np.linspace(0.0, xmax, bins + 1)

    ax_hist.hist(real, bins=bins_edges, density=True, alpha=0.6, label="Real")
    ax_hist.hist(gen,  bins=bins_edges, density=True, alpha=0.6, histtype="step", linewidth=1.8, label="Gen")
    ax_hist.set_xlabel(xlab); ax_hist.set_ylabel("density")
    ax_hist.set_title(title_left, fontsize=11, pad=6)
    ax_hist.legend(frameon=False, fontsize=9); ax_hist.grid(alpha=0.2, linestyle=":")
    _fmt_sci(ax_hist, "x"); ax_hist.tick_params(axis="both", labelsize=9)

    xr, yr = ecdf_tail_1m(real); xg, yg = ecdf_tail_1m(gen)
    ax_tail.plot(xr, yr, label="Real"); ax_tail.plot(xg, yg, linestyle="--", label="Gen")
    ax_tail.set_yscale("log")
    ax_tail.set_xlabel(xlab); ax_tail.set_ylabel("1-CDF")
    ax_tail.set_title("Tail (1−CDF)", fontsize=11, pad=6)
    ax_tail.legend(frameon=False, fontsize=9); ax_tail.grid(alpha=0.2, which="both", linestyle=":")
    _fmt_sci(ax_tail, "x"); ax_tail.tick_params(axis="both", labelsize=9)

def hist2d_with_colorbar(ax, x, y, bins=(80, 80), xrange=None, yrange=None,
                         title: str = "", xlabel: str = "", ylabel: str = "",
                         pixel_axes: bool = False, mid_lines: tuple[float,float] | None = None):
    h = ax.hist2d(x, y, bins=bins, range=[xrange, yrange])
    ax.set_title(title, fontsize=11, pad=6)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if pixel_axes: _fmt_pixel_axis(ax, "both")
    else: _fmt_sci(ax, "both")
    ax.tick_params(axis="both", labelsize=9)
    if mid_lines is not None:
        mx, my = mid_lines
        ax.axvline(mx, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(my, color="red", linestyle="--", linewidth=1.0)
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="3.5%", pad=0.06)
    cb = plt.colorbar(h[3], cax=cax); cb.set_label("counts", fontsize=9); cb.ax.tick_params(labelsize=8)

def hist1d_overlay(ax, real, gen, bins_edges, xlabel: str, title: str,
                   pixel_axis: bool = False, mid_line: float | None = None):
    ax.hist(real, bins=bins_edges, density=True, alpha=0.6, label="Real")
    ax.hist(gen,  bins=bins_edges, density=True, alpha=0.6, histtype="step", linewidth=1.8, label="Gen")
    ax.set_xlabel(xlabel); ax.set_ylabel("density")
    ax.set_title(title, fontsize=11, pad=6)
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.2, linestyle=":")
    if pixel_axis: _fmt_pixel_axis(ax, "x")
    else: _fmt_sci(ax, "x")
    if mid_line is not None:
        ax.axvline(mid_line, color="black", linestyle="--", linewidth=1.0)
    ax.tick_params(axis="both", labelsize=9)

def imshow_with_individual_colorbar(ax, img2d, title: str, xlabel: str, ylabel: str,
                                    pixel_axes: bool, mid_lines: tuple[float,float] | None = None,
                                    cmap="viridis", cbar_label: str = "counts"):
    im = ax.imshow(img2d, origin="lower", interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if pixel_axes: _fmt_pixel_axis(ax, "both")
    else: _fmt_sci(ax, "both")
    ax.tick_params(axis="both", labelsize=9)
    if mid_lines is not None:
        mx, my = mid_lines
        ax.axvline(mx, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(my, color="red", linestyle="--", linewidth=1.0)
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="3.5%", pad=0.06)
    cb = plt.colorbar(im, cax=cax); cb.set_label(cbar_label, fontsize=9); cb.ax.tick_params(labelsize=8)

# ---------- util: lettura colonne immagine ----------
def _stack_images(col) -> np.ndarray:
    # col è una Series di liste o array length H*W
    mats = [np.asarray(v, dtype=np.float32) for v in col]
    arr = np.vstack(mats)
    return arr.reshape(-1, H, W)

# ---------- centratura ----------
def center_images_fft(imgs: np.ndarray, bx01: np.ndarray, by01: np.ndarray,
                      device: torch.device | None = None) -> np.ndarray:
    """
    imgs: [N,H,W], bx01/by01 in [0,1]
    ritorna imgs centrate con baricentro messo in (MID_X, MID_Y)
    """
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
    ap.add_argument("--bins-bary-x", type=int, default=W, help="Bin istogramma 1D barycenter X (pixel)")
    ap.add_argument("--bins-bary-y", type=int, default=H, help="Bin istogramma 1D barycenter Y (pixel)")
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

    # --- vettori scalari
    psum_true = df["psum_true"].to_numpy(np.float64, copy=False)
    psum_gen  = df["psum_gen"].to_numpy(np.float64, copy=False)
    pmax_true = df["pmax_true"].to_numpy(np.float64, copy=False)
    pmax_gen  = df["pmax_gen"].to_numpy(np.float64, copy=False)

    # ============ Canvas 1: photon sum ============
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=False)
    dual_hist_and_tail(axs[0], axs[1], psum_true, psum_gen,
                       title_left="Histogram overlap",
                       xlab="counts", bins=args.bins)
    save_figure(fig, out_dir / "psum_comparison.png", "photon sum comparison", args.dpi)

    # ============ Canvas 2: pmax ============
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=False)
    dual_hist_and_tail(axs[0], axs[1], pmax_true, pmax_gen,
                       title_left="Histogram overlap",
                       xlab="counts", bins=args.bins)
    save_figure(fig, out_dir / "pmax_comparison.png", "pmax comparison", args.dpi)

    # ============ Canvas 3: 2D hist (psum vs pmax) ============
    x_max = float(max(np.nanmax(psum_true), np.nanmax(psum_gen))); x_max = 1.0 if not np.isfinite(x_max) else max(1.0, x_max)
    y_max = float(max(np.nanmax(pmax_true), np.nanmax(pmax_gen))); y_max = 1.0 if not np.isfinite(y_max) else max(1.0, y_max)
    xrange = (0.0, x_max); yrange = (0.0, y_max)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=False)
    hist2d_with_colorbar(axs[0], psum_true, pmax_true, bins=(args.bins2d, args.bins2d),
                         xrange=xrange, yrange=yrange,
                         title="Real: 2D histogram", xlabel="psum (counts)", ylabel="pmax (counts)",
                         pixel_axes=False, mid_lines=None)
    hist2d_with_colorbar(axs[1], psum_gen,  pmax_gen,  bins=(args.bins2d, args.bins2d),
                         xrange=xrange, yrange=yrange,
                         title="Generated: 2D histogram", xlabel="psum (counts)", ylabel="pmax (counts)",
                         pixel_axes=False, mid_lines=None)
    save_figure(fig, out_dir / "psum_pmax_2D.png", "psum vs pmax — Real vs Generated", args.dpi)

    # ============ Canvas 4: Impact point distributions (2x2) ============
    bx_t_px = (df["bar_x_true"].to_numpy(np.float64, copy=False)) * (W - 1.0)
    by_t_px = (df["bar_y_true"].to_numpy(np.float64, copy=False)) * (H - 1.0)
    bx_g_px = (df["bar_x_gen"].to_numpy(np.float64,  copy=False)) * (W - 1.0)
    by_g_px = (df["bar_y_gen"].to_numpy(np.float64,  copy=False)) * (H - 1.0)

    fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.6), constrained_layout=False)
    xy_range = ([0.0, W - 1.0], [0.0, H - 1.0]); bins2d_bary = (args.bins2d, args.bins2d)
    hist2d_with_colorbar(axs[0,0], bx_t_px, by_t_px, bins=bins2d_bary,
                         xrange=xy_range[0], yrange=xy_range[1],
                         title="Real: impact 2D histogram",
                         xlabel="barycenter x (pixel)", ylabel="barycenter y (pixel)",
                         pixel_axes=True, mid_lines=(MID_X, MID_Y))
    hist2d_with_colorbar(axs[0,1], bx_g_px, by_g_px, bins=bins2d_bary,
                         xrange=xy_range[0], yrange=xy_range[1],
                         title="Generated: impact 2D histogram",
                         xlabel="barycenter x (pixel)", ylabel="barycenter y (pixel)",
                         pixel_axes=True, mid_lines=(MID_X, MID_Y))
    x_edges = np.linspace(0.0, W - 1.0, int(args.bins_bary_x) + 1)
    y_edges = np.linspace(0.0, H - 1.0, int(args.bins_bary_y) + 1)
    hist1d_overlay(axs[1,0], bx_t_px, bx_g_px, x_edges,
                   xlabel="barycenter x (pixel)", title="Histogram overlap — x",
                   pixel_axis=True, mid_line=MID_X)
    hist1d_overlay(axs[1,1], by_t_px, by_g_px, y_edges,
                   xlabel="barycenter y (pixel)", title="Histogram overlap — y",
                   pixel_axis=True, mid_line=MID_Y)
    save_figure(fig, out_dir / "impact_point_2x2.png", "impact point distributions", args.dpi)

    # ============ Canvas 5: OCCUPANCY (centered) sulle FINAL ============
    finals_real = _stack_images(df["final_real"])
    finals_gen  = _stack_images(df["final_gen"])

    bx_t01 = df["bar_x_true"].to_numpy(np.float32, copy=False)
    by_t01 = df["bar_y_true"].to_numpy(np.float32, copy=False)
    bx_g01 = df["bar_x_gen"].to_numpy(np.float32,  copy=False)
    by_g01 = df["bar_y_gen"].to_numpy(np.float32,  copy=False)

    finals_real_c = center_images_fft(finals_real, bx_t01, by_t01)
    finals_gen_c  = center_images_fft(finals_gen,  bx_g01, by_g01)

    # occupancy: +1 se pixel >= 1, altrimenti 0
    occ_real = (finals_real_c >= 1.0).sum(axis=0).astype(np.float64)
    occ_gen  = (finals_gen_c  >= 1.0).sum(axis=0).astype(np.float64)

    # proiezioni
    projx_real = occ_real.sum(axis=0); projx_gen = occ_gen.sum(axis=0)
    projy_real = occ_real.sum(axis=1); projy_gen = occ_gen.sum(axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(12.2, 8.8), constrained_layout=False)
    imshow_with_individual_colorbar(axs[0,0], occ_real, "Real — occupancy (centered, final)",
                                    "x (pixel)", "y (pixel)", pixel_axes=True,
                                    mid_lines=(MID_X, MID_Y), cbar_label="counts")
    imshow_with_individual_colorbar(axs[0,1], occ_gen,  "Generated — occupancy (centered, final)",
                                    "x (pixel)", "y (pixel)", pixel_axes=True,
                                    mid_lines=(MID_X, MID_Y), cbar_label="counts")
    # profili
    xs = np.arange(W); ys = np.arange(H)
    axs[1,0].plot(xs, projx_real, label="Real"); axs[1,0].plot(xs, projx_gen, linestyle="--", label="Gen")
    axs[1,0].set_title("X projection", fontsize=11, pad=6)
    axs[1,0].set_xlabel("x (pixel)"); axs[1,0].set_ylabel("counts")
    axs[1,0].legend(frameon=False, fontsize=9); axs[1,0].grid(alpha=0.2, linestyle=":")
    _fmt_pixel_axis(axs[1,0], "x"); axs[1,0].axvline(MID_X, color="black", linestyle="--", linewidth=1.0)

    axs[1,1].plot(ys, projy_real, label="Real"); axs[1,1].plot(ys, projy_gen, linestyle="--", label="Gen")
    axs[1,1].set_title("Y projection", fontsize=11, pad=6)
    axs[1,1].set_xlabel("y (pixel)"); axs[1,1].set_ylabel("counts")
    axs[1,1].legend(frameon=False, fontsize=9); axs[1,1].grid(alpha=0.2, linestyle=":")
    _fmt_pixel_axis(axs[1,1], "x"); axs[1,1].axvline(MID_Y, color="black", linestyle="--", linewidth=1.0)

    save_figure(fig, out_dir / "centered_occupancy_final_2x2.png",
                "centered occupancy maps (final) & projections", args.dpi)

    # ============ Canvas 6: SUM (centered) sulle FINAL ============
    sum_real = finals_real_c.sum(axis=0).astype(np.float64)
    sum_gen  = finals_gen_c.sum(axis=0).astype(np.float64)

    projx_real_s = sum_real.sum(axis=0); projx_gen_s = sum_gen.sum(axis=0)
    projy_real_s = sum_real.sum(axis=1); projy_gen_s = sum_gen.sum(axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(12.2, 8.8), constrained_layout=False)
    imshow_with_individual_colorbar(axs[0,0], sum_real, "Real — sum (centered, final)",
                                    "x (pixel)", "y (pixel)", pixel_axes=True,
                                    mid_lines=(MID_X, MID_Y), cbar_label="sum")
    imshow_with_individual_colorbar(axs[0,1], sum_gen,  "Generated — sum (centered, final)",
                                    "x (pixel)", "y (pixel)", pixel_axes=True,
                                    mid_lines=(MID_X, MID_Y), cbar_label="sum")
    # profili
    axs[1,0].plot(xs, projx_real_s, label="Real"); axs[1,0].plot(xs, projx_gen_s, linestyle="--", label="Gen")
    axs[1,0].set_title("X projection (sum)", fontsize=11, pad=6)
    axs[1,0].set_xlabel("x (pixel)"); axs[1,0].set_ylabel("sum")
    axs[1,0].legend(frameon=False, fontsize=9); axs[1,0].grid(alpha=0.2, linestyle=":")
    _fmt_pixel_axis(axs[1,0], "x"); axs[1,0].axvline(MID_X, color="black", linestyle="--", linewidth=1.0)

    axs[1,1].plot(ys, projy_real_s, label="Real"); axs[1,1].plot(ys, projy_gen_s, linestyle="--", label="Gen")
    axs[1,1].set_title("Y projection (sum)", fontsize=11, pad=6)
    axs[1,1].set_xlabel("y (pixel)"); axs[1,1].set_ylabel("sum")
    axs[1,1].legend(frameon=False, fontsize=9); axs[1,1].grid(alpha=0.2, linestyle=":")
    _fmt_pixel_axis(axs[1,1], "x"); axs[1,1].axvline(MID_Y, color="black", linestyle="--", linewidth=1.0)

    save_figure(fig, out_dir / "centered_sum_final_2x2.png",
                "centered sum maps (final) & projections", args.dpi)

    # ============ Canvas 7: Random samples (FINAL) ============
    n_total = finals_real.shape[0]
    n_show = min(4, n_total)
    sel = np.random.choice(n_total, size=n_show, replace=False)

    fig, axs = plt.subplots(2, 4, figsize=(12.8, 6.8), constrained_layout=False)
    for k, idx in enumerate(sel):
        r = finals_real[idx]
        g = finals_gen[idx]
        row = 0 if k < 2 else 1
        col_pair = (k % 2) * 2

        ax_r = axs[row, col_pair]
        ax_g = axs[row, col_pair + 1]

        im_r = ax_r.imshow(r, origin="lower", interpolation="nearest", cmap="viridis")
        ax_r.set_title(f"Final Real #{k+1} (idx {int(idx)})", fontsize=10, pad=4)
        _fmt_pixel_axis(ax_r, "both")
        ax_r.set_xlabel("x (pixel)"); ax_r.set_ylabel("y (pixel)")
        div_r = make_axes_locatable(ax_r); cax_r = div_r.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im_r, cax=cax_r).ax.tick_params(labelsize=8)

        im_g = ax_g.imshow(g, origin="lower", interpolation="nearest", cmap="viridis")
        ax_g.set_title(f"Final Gen #{k+1} (idx {int(idx)})", fontsize=10, pad=4)
        _fmt_pixel_axis(ax_g, "both")
        ax_g.set_xlabel("x (pixel)"); ax_g.set_ylabel("y (pixel)")
        div_g = make_axes_locatable(ax_g); cax_g = div_g.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im_g, cax=cax_g).ax.tick_params(labelsize=8)

    save_figure(fig, out_dir / "samples_final_real_vs_gen.png",
                "random samples (final) — real vs generated", args.dpi)

    # ============ Canvas 8: Random samples (SHAPE) ============
    shapes_real = _stack_images(df["shape_real"])
    shapes_gen  = _stack_images(df["shape_gen"])

    n_total_s = shapes_real.shape[0]
    n_show_s = min(4, n_total_s)
    sel_s = np.random.choice(n_total_s, size=n_show_s, replace=False)

    fig, axs = plt.subplots(2, 4, figsize=(12.8, 6.8), constrained_layout=False)
    for k, idx in enumerate(sel_s):
        r = shapes_real[idx]
        g = shapes_gen[idx]
        row = 0 if k < 2 else 1
        col_pair = (k % 2) * 2

        ax_r = axs[row, col_pair]
        ax_g = axs[row, col_pair + 1]

        im_r = ax_r.imshow(r, origin="lower", interpolation="nearest", cmap="viridis")
        ax_r.set_title(f"Shape Real #{k+1} (idx {int(idx)})", fontsize=10, pad=4)
        _fmt_pixel_axis(ax_r, "both")
        ax_r.set_xlabel("x (pixel)"); ax_r.set_ylabel("y (pixel)")
        div_r = make_axes_locatable(ax_r); cax_r = div_r.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im_r, cax=cax_r).ax.tick_params(labelsize=8)

        im_g = ax_g.imshow(g, origin="lower", interpolation="nearest", cmap="viridis")
        ax_g.set_title(f"Shape Gen #{k+1} (idx {int(idx)})", fontsize=10, pad=4)
        _fmt_pixel_axis(ax_g, "both")
        ax_g.set_xlabel("x (pixel)"); ax_g.set_ylabel("y (pixel)")
        div_g = make_axes_locatable(ax_g); cax_g = div_g.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im_g, cax=cax_g).ax.tick_params(labelsize=8)

    save_figure(fig, out_dir / "samples_shape_real_vs_gen.png",
                "random samples (shape) — real vs generated", args.dpi)

    print(f"[OK] Salvate 8 figure in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# ============================================================
# dataset_builder.py — EDA con 1/N dN/dx (TeX labels) + correlazioni + 4 immagini
# - Canvas A (2x3): E, pz, px, py, pT, |p|
# - Canvas B (2x3): [vx, vy, vz] + [corr (vx,vy), corr (vx,vz), corr (vy,vz)]
# - Canvas C (4x1): 4 immagini (ognuna con colorbar propria, z-range indipendente)
# Lettura immagini:
#   1) Se disponibile LMDB per lo shard (da manifest), usa LMDB
#   2) Altrimenti fallback al pickle sorgente e alla colonna immagine indicata nel manifest
# ============================================================

from __future__ import annotations
import argparse
from pathlib import Path
import gzip, json, random
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ------------------------ Colormap ROOT-like (zero trasparente) ------------------------
def root_like_colormap() -> LinearSegmentedColormap:
    colors = [
        (0.00, (0.05, 0.05, 0.35)),  # deep blue
        (0.25, (0.00, 0.45, 0.80)),  # blue-cyan
        (0.50, (0.05, 0.70, 0.50)),  # teal-green
        (0.75, (0.60, 0.85, 0.20)),  # green-yellow
        (1.00, (0.98, 0.92, 0.15)),  # warm yellow
    ]
    cmap = LinearSegmentedColormap.from_list("rootlike", colors)
    cmap.set_bad(alpha=0.0)  # zeri trasparenti
    return cmap

# ------------------------ Helpers label stile TeX ------------------------
def _fmt_no_sci(ax, which: str = "both"):
    def _mk():
        fmt = ScalarFormatter(useMathText=False)
        fmt.set_scientific(False)
        return fmt
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(_mk())
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(_mk())

def x_label_tex(symb: str, unit_tex: str) -> str:
    # unit_tex es: r"\mathrm{GeV}", r"\mathrm{GeV}/c", r"\mathrm{m}"
    return rf"${symb}\;[{unit_tex}]$"

def y_label_tex(symb: str) -> str:
    # 1/N dN/d(symb)
    return rf"$\frac{{1}}{{N}}\,\mathrm{{d}}N/\mathrm{{d}}{symb}$"

# ------------------------ IO: compact parquet + manifest ------------------------
NEEDED_COLS = ["E", "px", "py", "pz", "vx", "vy", "vz"]

def collect_parquet_paths(root: Path, splits: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for sp in splits:
        d = root / f"{sp}_compact"
        if d.is_dir():
            paths.extend(sorted(d.glob("*.parquet")))
    return paths

def read_columns(paths: List[Path], columns: List[str]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("Nessun file .parquet trovato.")
    dfs = [pd.read_parquet(p, columns=columns) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def load_manifest(stats_dir: Path) -> dict:
    mpath = stats_dir / "manifest.json.gz"
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest non trovato: {mpath}")
    with gzip.open(mpath, "rt") as f:
        return json.load(f)

# ------------------------ Lettura immagini (LMDB o fallback pickle) ------------------------
def _read_images_from_lmdb(lmdb_path: Path, idx_list: List[int]) -> List[np.ndarray]:
    import lmdb
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=True, max_readers=2048)
    imgs: List[np.ndarray] = []
    with env.begin(write=False) as txn:
        for i in idx_list:
            key = f"{i:08d}".encode("ascii")  # stesso formato writer standard
            blob = txn.get(key)
            if blob is None:
                continue
            arr = np.frombuffer(blob, dtype=np.float32)
            # l'H e W precisi li ricaviamo fuori e facciamo reshape lì
            imgs.append(arr)
    env.close()
    return imgs

def _read_images_from_pickle(pickle_path: Path, image_col: str, idx_list: List[int]) -> List[np.ndarray]:
    df = pd.read_pickle(pickle_path)
    out = []
    for i in idx_list:
        im = np.asarray(df.iloc[i][image_col])
        if im.ndim == 3 and im.shape[0] == 1:
            im = im[0]
        out.append(im.astype(np.float32, copy=False))
    return out

def sample_four_images(manifest: dict, prefer_split: str = "train") -> Tuple[List[np.ndarray], int, int]:
    """
    Prova a prendere 4 immagini da:
      - uno shard train (preferito) con LMDB; se non c'è, fallback a pickle
      - altrimenti test
    Ritorna (lista immagini 2D float32, H, W).
    """
    rng = random.Random(42)
    entries = []
    if prefer_split in manifest and manifest[prefer_split]:
        if prefer_split == "train":
            entries.extend(manifest["train"])
        else:
            entries.append(manifest["test"])
    # se vuoto, prova l'altro split
    if not entries:
        other = "test" if prefer_split == "train" else "train"
        if other in manifest and manifest[other]:
            if other == "train":
                entries.extend(manifest["train"])
            else:
                entries.append(manifest["test"])
    if not entries:
        raise RuntimeError("Manifest privo di riferimenti a shard train/test con immagini.")

    rng.shuffle(entries)
    for meta in entries:
        H = int(meta["H"]); W = int(meta["W"])
        n_rows = int(meta["rows"])
        idx = sorted(set(rng.sample(range(n_rows), k=min(4, n_rows))))
        # prova LMDB
        lmdb_path = meta.get("lmdb")
        if lmdb_path:
            lmdb_path = Path(lmdb_path)
            try:
                vecs = _read_images_from_lmdb(lmdb_path, idx)
                if len(vecs) == len(idx):
                    imgs = [v.reshape(H, W) for v in vecs]
                    return imgs, H, W
            except Exception:
                pass
        # fallback: shard sorgente pickle
        shard_src = Path(meta["shard_src"])
        img_col = meta.get("image_col", "image")
        try:
            ims = _read_images_from_pickle(shard_src, img_col, idx)
            if len(ims) == len(idx):
                return ims, H, W
        except Exception:
            continue

    raise RuntimeError("Impossibile leggere 4 immagini (né LMDB né pickle).")

# ------------------------ Istogrammi normalizzati ------------------------
def normalized_hist(ax: plt.Axes, data: np.ndarray, bins: int, xlab_tex: str, title_tex: str,
                    symb_tex: str, clip_to: Optional[Tuple[float,float]] = None):
    x = np.asarray(data, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return

    if clip_to is None:
        lo, hi = np.quantile(x, [1e-3, 1 - 1e-3])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(x)), float(np.max(x))
    else:
        lo, hi = clip_to

    edges = np.linspace(lo, hi, int(bins) + 1)
    counts, edges = np.histogram(x, bins=edges, density=False)
    widths = np.diff(edges)
    N = x.size
    y = counts / (N * widths)
    centers = edges[:-1] + widths * 0.5

    ax.step(centers, y, where="mid", linewidth=1.6)
    ax.set_title(rf"${title_tex}$", fontsize=11, pad=6)
    ax.set_xlabel(xlab_tex)
    ax.set_ylabel(y_label_tex(symb_tex))
    ax.grid(alpha=0.25, linestyle=":")
    ax.tick_params(axis="both", labelsize=9)
    _fmt_no_sci(ax, "both")
    ax.set_xlim(lo, hi)

# ------------------------ Correlazioni 2D ------------------------
def corr2d(ax: plt.Axes, x: np.ndarray, y: np.ndarray, bins2d: int,
           xrange: Tuple[float,float], yrange: Tuple[float,float],
           title_tex: str, xlab_tex: str, ylab_tex: str):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return None

    H, xe, ye = np.histogram2d(x, y, bins=(bins2d, bins2d), range=[xrange, yrange])
    data = np.ma.masked_less_equal(H.T, 0.0)
    cmap = root_like_colormap()
    vmax = float(max(1.0, H.max()))
    im = ax.pcolormesh(xe, ye, data, shading="flat", cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax))
    ax.set_title(rf"${title_tex}$", fontsize=11, pad=6)
    ax.set_xlabel(xlab_tex); ax.set_ylabel(ylab_tex)
    ax.tick_params(axis="both", labelsize=9)
    _fmt_no_sci(ax, "both")
    ax.set_xlim(*xrange); ax.set_ylim(*yrange)
    return im

def add_cbar(ax: plt.Axes, mappable, label: str = "Counts", size: str = "4%", pad: float = 0.04):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = ax.figure.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb

# ------------------------ Immagini (4 in colonna) ------------------------
def plot_images_column(imgs: List[np.ndarray], out_path: Path, dpi: int):
    fig_h = 1.0 + 2.8 * len(imgs)
    fig, axs = plt.subplots(len(imgs), 1, figsize=(6.2, fig_h), constrained_layout=False)
    if len(imgs) == 1:
        axs = [axs]
    cmap = root_like_colormap()

    for k, (ax, im2d) in enumerate(zip(axs, imgs)):
        im2d = np.asarray(im2d, dtype=np.float64)
        data = np.ma.masked_less_equal(im2d, 0.0)
        vmax = float(np.nanmax(im2d)); 
        if not np.isfinite(vmax) or vmax <= 0: vmax = 1.0
        m = ax.imshow(data, origin="lower", interpolation="nearest",
                      cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax))
        ax.set_title(rf"$\mathrm{{Image}}\ #{k+1}$", fontsize=11, pad=6)
        ax.set_xlabel(r"$x\ [\mathrm{pixel}]$"); ax.set_ylabel(r"$y\ [\mathrm{pixel}]$")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        _fmt_no_sci(ax, "both")
        add_cbar(ax, m, label="counts", size="5%", pad=0.08)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ------------------------ Main ------------------------
def main():
    ap = argparse.ArgumentParser(description="EDA ZDC: 1/N dN/dx + correlazioni + immagini (compatibile con manifest)")
    ap.add_argument("--data-root", default="/data/dataalice/dfuligno/ZDC_fastsim/data/",
                    help="Cartella che contiene train_compact/, test_compact/ e stats/")
    ap.add_argument("--splits", default="train,test",
                    help="Quali split usare per i Parquet (comma-separated)")
    ap.add_argument("--out-dir", default="eda", help="Cartella di output figure")
    ap.add_argument("--bins", type=int, default=120, help="Bin 1D")
    ap.add_argument("--bins2d", type=int, default=140, help="Bin per lato (2D)")
    ap.add_argument("--dpi", type=int, default=170, help="DPI PNG")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Lettura variabili scalari (Parquet compact) ----
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    paths = collect_parquet_paths(root, splits)
    if not paths:
        raise SystemExit(f"Nessun .parquet trovato in {root} per gli split {splits}.")
    df = read_columns(paths, NEEDED_COLS)

    # ---- Canvas A: 2x3 kinematics ----
    px = df["px"].to_numpy(np.float64, copy=False)
    py = df["py"].to_numpy(np.float64, copy=False)
    pz = df["pz"].to_numpy(np.float64, copy=False)
    E  = df["E"].to_numpy(np.float64, copy=False)
    pT = np.sqrt(px**2 + py**2)
    p  = np.sqrt(px**2 + py**2 + pz**2)

    fig, axs = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=False)
    panels = [
        (E,  x_label_tex(r"E", r"\mathrm{GeV}"),          r"E",   r"E"),
        (pz, x_label_tex(r"p_z", r"\mathrm{GeV}/c"),      r"p_z", r"p_z"),
        (px, x_label_tex(r"p_x", r"\mathrm{GeV}/c"),      r"p_x", r"p_x"),
        (py, x_label_tex(r"p_y", r"\mathrm{GeV}/c"),      r"p_y", r"p_y"),
        (pT, x_label_tex(r"p_T", r"\mathrm{GeV}/c"),      r"p_T", r"p_T"),
        (p,  x_label_tex(r"|p|", r"\mathrm{GeV}/c"),      r"|p|", r"|p|"),
    ]
    for ax, (data, xlab, title, symb) in zip(axs.ravel(), panels):
        normalized_hist(ax, data, bins=args.bins, xlab_tex=xlab, title_tex=title, symb_tex=symb)

    fig.tight_layout()
    outA = out_dir / "eda_canvas_A_kinematics.png"
    fig.savefig(outA, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Salvato: {outA}")

    # ---- Canvas B: v's + correlazioni 2D (con limiti fissi) ----
    vx = df["vx"].to_numpy(np.float64, copy=False)
    vy = df["vy"].to_numpy(np.float64, copy=False)
    vz = df["vz"].to_numpy(np.float64, copy=False)
    vlim = (-0.05, 0.05)

    fig = plt.figure(figsize=(13.6, 8.6))
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1.0, 1.25], wspace=0.36, hspace=0.36)

    # Riga 1: istogrammi normalizzati con limiti fissi
    ax_vx = fig.add_subplot(gs[0, 0]); normalized_hist(ax_vx, vx, args.bins, x_label_tex(r"v_x", r"\mathrm{m}"), r"v_x", r"v_x", clip_to=vlim)
    ax_vy = fig.add_subplot(gs[0, 1]); normalized_hist(ax_vy, vy, args.bins, x_label_tex(r"v_y", r"\mathrm{m}"), r"v_y", r"v_y", clip_to=vlim)
    ax_vz = fig.add_subplot(gs[0, 2]); normalized_hist(ax_vz, vz, args.bins, x_label_tex(r"v_z", r"\mathrm{m}"), r"v_z", r"v_z", clip_to=vlim)

    # Riga 2: tre correlazioni 2D con colorbar dedicata
    ax_c1 = fig.add_subplot(gs[1, 0])
    im1 = corr2d(ax_c1, vx, vy, bins2d=args.bins2d, xrange=vlim, yrange=vlim,
                 title_tex=r"\mathrm{Corr}\ (v_x,\ v_y)", xlab_tex=x_label_tex(r"v_x", r"\mathrm{m}"),
                 ylab_tex=x_label_tex(r"v_y", r"\mathrm{m}"))
    if im1 is not None: add_cbar(ax_c1, im1, label="Counts", size="4.5%", pad=0.06)

    ax_c2 = fig.add_subplot(gs[1, 1])
    im2 = corr2d(ax_c2, vx, vz, bins2d=args.bins2d, xrange=vlim, yrange=vlim,
                 title_tex=r"\mathrm{Corr}\ (v_x,\ v_z)", xlab_tex=x_label_tex(r"v_x", r"\mathrm{m}"),
                 ylab_tex=x_label_tex(r"v_z", r"\mathrm{m}"))
    if im2 is not None: add_cbar(ax_c2, im2, label="Counts", size="4.5%", pad=0.06)

    ax_c3 = fig.add_subplot(gs[1, 2])
    im3 = corr2d(ax_c3, vy, vz, bins2d=args.bins2d, xrange=vlim, yrange=vlim,
                 title_tex=r"\mathrm{Corr}\ (v_y,\ v_z)", xlab_tex=x_label_tex(r"v_y", r"\mathrm{m}"),
                 ylab_tex=x_label_tex(r"v_z", r"\mathrm{m}"))
    if im3 is not None: add_cbar(ax_c3, im3, label="Counts", size="4.5%", pad=0.06)

    outB = out_dir / "eda_canvas_B_vertices_corr.png"
    fig.savefig(outB, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Salvato: {outB}")

    # ---- Canvas C: 4 immagini in colonna (LMDB → fallback pickle) ----
    try:
        manifest = load_manifest(root / "stats")
        imgs, H, W = sample_four_images(manifest, prefer_split="train")
        outC = out_dir / "eda_canvas_C_images.png"
        plot_images_column(imgs, outC, dpi=args.dpi)
        print(f"[OK] Salvato: {outC}")
    except Exception as e:
        print(f"[WARN] Canvas C (immagini) saltata: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Stage B — Conditional Flow Matching per shape 44x44 centrata e normalizzata
from __future__ import annotations
import os, math, time, argparse, json, gzip
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------- Costanti/colonne --------------------
H = 44
W = 44
COND_COLS_DEFAULT = ["theta","phi","ux","uy","uz","E"]  # 6 cond feature consigliate

# -------------------- Utils IO --------------------
def load_json_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Warp/Subpixel shift --------------------
def fourier_shift_2d(img: torch.Tensor, tx: float, ty: float) -> torch.Tensor:
    """
    img: [H,W] (float32)
    tx, ty in pixel (float): +x = destra (colonne), +y = giù (righe)
    Usa phase shift in Fourier per traslazione sub-pixel.
    """
    Fimg = torch.fft.rfft2(img)
    yy = torch.fft.fftfreq(img.shape[0], d=1.0, device=img.device)   # H
    xx = torch.fft.rfftfreq(img.shape[1], d=1.0, device=img.device)  # W/2+1
    phase = torch.exp(-2j*math.pi*(yy[:,None]*ty + xx[None,:]*tx))
    shifted = torch.fft.irfft2(Fimg * phase, s=img.shape)
    return shifted

# -------------------- Dataset --------------------
class ShapeCFMDataset(Dataset):
    """
    Carica righe dal compact parquet *e* le immagini:
      - preferisce LMDB se presente nel manifest
      - se LMDB assente, fallback allo shard .pkl/.plk originale (stesso ordine righe)
    Genera shape centrata e normalizzata (somma=1) usando di default x_imp,y_imp,T dal dataset;
    con --recompute-from-image ricalcola da immagine.
    """
    def __init__(self,
                 rows_meta: List[Dict[str, Any]],
                 cond_cols: List[str],
                 recompute_from_image: bool = False):
        """
        rows_meta: lista di dict, ciascuno:
          {
            "parquet": path al parquet,
            "lmdb": path lmdb oppure None,
            "shard_src": path allo shard pkl/plk,
            "image_col": nome colonna immagine nello shard,
          }
        """
        super().__init__()
        self.cond_cols = cond_cols
        self.recompute = recompute_from_image

        # carica tutti i parquet e memorizza mapping globale→locale
        self.parquets: List[pd.DataFrame] = []
        self.global_to_local: List[Tuple[int,int]] = []  # (pid, rid)
        for meta in rows_meta:
            pq = pd.read_parquet(meta["parquet"])
            pid = len(self.parquets)
            self.parquets.append(pq)
            n = len(pq)
            for rid in range(n):
                self.global_to_local.append((pid, rid))

        # info IO per ogni parquet
        self.per_pq_io = []
        for meta in rows_meta:
            self.per_pq_io.append({
                "lmdb": meta.get("lmdb", None),
                "shard_src": meta.get("shard_src", None),
                "image_col": meta.get("image_col", None),
            })

        # cache shard pkl (fallback)
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

        # 1) immagine (LMDB se presente, altrimenti shard_src pkl)
        if io["lmdb"]:
            img_np = self._read_image_from_lmdb(io["lmdb"], rid)
        elif io["shard_src"] and io["image_col"]:
            img_np = self._read_image_from_pkl(io["shard_src"], io["image_col"], rid)
        else:
            raise RuntimeError(
                "Impossibile leggere immagine: né LMDB né shard_src disponibili. "
                "Rigenera i compact con --lmdb oppure verifica il manifest."
            )

        # 2) centro e normalizzazione
        if self.recompute:
            total = float(img_np.sum())
            if total <= 0:
                cx_pix = (W-1)/2.0; cy_pix = (H-1)/2.0
            else:
                ys, xs = np.indices((H,W))
                cx_pix = float((xs*img_np).sum()/(total+1e-12))
                cy_pix = float((ys*img_np).sum()/(total+1e-12))
        else:
            # usa x_imp,y_imp,T dal dataset — sono in [0,1] ⇒ converti in pixel-space
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

        # 3) cond features
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

# -------------------- CFM loss --------------------
def cfm_loss(model, x0, c):
    """
    Conditional flow matching:
      x_t = (1-t) x0 + t x1, con x1 ~ N(0, I).
      dx/dt = x1 - x0.
      v_theta ≈ E[ || v_theta(x_t, t, c) - (x1 - x0) ||^2 ].
    """
    B = x0.size(0)
    t = torch.rand(B, device=x0.device).view(B,1,1,1)  # [B,1,1,1]
    x1 = torch.randn_like(x0)
    xt = (1.0 - t) * x0 + t * x1
    target = x1 - x0
    pred = model(xt, t.view(B,1), c)
    return F.mse_loss(pred, target)

# -------------------- CSV logger --------------------
def append_csv(path: Path, row: Dict[str, Any], header: Optional[List[str]] = None):
    """
    Se il file non esiste e header è fornito, crea l'header senza righe.
    Poi, se 'row' non è vuoto, appende la riga.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not path.exists()) and header is not None:
        with open(path, "w") as f:
            pd.DataFrame(columns=header).to_csv(f, header=True, index=False)
    if row:
        with open(path, "a") as f:
            pd.DataFrame([row]).to_csv(f, header=False, index=False)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Stage B — Conditional Flow Matching (shape 44x44)")
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/",
                    help="Cartella con train_compact/, test_compact/, stats/ (e manifest.json.gz)")
    ap.add_argument("--out-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_B",
                    help="Dove salvare ckpt e CSV")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--cond-cols", type=str, default=",".join(COND_COLS_DEFAULT),
                    help="Colonne condizionali (comma-sep) prese dal compact parquet")
    ap.add_argument("--recompute-from-image", action="store_true",
                    help="Se presente, ricalcola (centro, totale) dalla matrice invece di usare (x_imp,y_imp,T) del dataset")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train-shards", type=int, default=1,
                    help="Usa i primi N shard di train dal manifest (<=0 per usarli tutti)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    compact_dir = Path(args.compact_dir)
    manifest_path = compact_dir / "stats" / "manifest.json.gz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}. Genera i compact con dataset_builder.py")

    manifest = load_json_gz(str(manifest_path))
    train_meta_all: List[Dict[str, Any]] = manifest["train"]
    test_meta  = manifest["test"]

    # seleziona quanti shard usare
    if args.train_shards is None:
        use_n = 1
    else:
        use_n = int(args.train_shards)
    if use_n <= 0 or use_n > len(train_meta_all):
        use_n = len(train_meta_all)

    train_meta = train_meta_all[:use_n]
    print(f"[INFO] Userò {use_n} shard di train su {len(train_meta_all)} totali.")

    # Prepara meta per dataset: parquet + (lmdb OR shard_src/image_col) per fallback
    rows_train = []
    for m in train_meta:
        rows_train.append({
            "parquet": m["parquet"],
            "lmdb": m.get("lmdb", None),
            "shard_src": m.get("shard_src", None),
            "image_col": m.get("image_col", None),
        })
    rows_test = [{
        "parquet": test_meta["parquet"],
        "lmdb": test_meta.get("lmdb", None),
        "shard_src": test_meta.get("shard_src", None),
        "image_col": test_meta.get("image_col", None),
    }]

    cond_cols = [s.strip() for s in args.cond_cols.split(",") if s.strip()]
    cond_dim = len(cond_cols)
    print(f"[INFO] Start Stage B CFM: epochs={args.epochs}  batch={args.batch_size}  cond_dim={cond_dim}  recompute_from_image={args.recompute_from_image}")

    ds_tr = ShapeCFMDataset(rows_train, cond_cols, recompute_from_image=args.recompute_from_image)
    ds_te = ShapeCFMDataset(rows_test,  cond_cols, recompute_from_image=args.recompute_from_image)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetCFM(cond_dim=cond_dim, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = out_dir / "stageB_cfm_best.pt"
    csv_path  = out_dir / "stageB_train_log.csv"

    best_val = float("inf")
    header = ["epoch","train_loss","val_loss","lr","time_sec","batch_size","cond_cols","recompute_from_image","train_shards_used"]
    append_csv(csv_path, {}, header=header)  # crea solo l'header

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        # ---- train ----
        model.train()
        run, seen = 0.0, 0
        for it, (x0, c) in enumerate(dl_tr):
            x0 = x0.to(device)
            c  = c.to(device)
            opt.zero_grad(set_to_none=True)
            loss = cfm_loss(model, x0, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.item()) * x0.size(0); seen += x0.size(0)
            if args.print_every and (it+1) % args.print_every == 0:
                print(f"  [ep {ep:03d}] it={it+1:05d} train_loss={run/seen:.5f}")
        train_loss = run / max(1,seen)

        # ---- val ----
        model.eval()
        v_run, v_seen = 0.0, 0
        with torch.no_grad():
            for x0, c in dl_te:
                x0 = x0.to(device); c = c.to(device)
                v = cfm_loss(model, x0, c)
                v_run += float(v.item()) * x0.size(0); v_seen += x0.size(0)
        val_loss = v_run / max(1, v_seen)
        dt = time.time() - t0
        print(f"[EP {ep:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({dt:.1f}s)")

        # CSV (append riga)
        append_csv(csv_path, {
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(opt.param_groups[0]["lr"]),
            "time_sec": float(dt),
            "batch_size": int(args.batch_size),
            "cond_cols": "|".join(cond_cols),
            "recompute_from_image": int(args.recompute_from_image),
            "train_shards_used": int(use_n),
        })

        # best
        if val_loss < best_val and np.isfinite(val_loss):
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "cond_cols": cond_cols,
                "H": H, "W": W,
            }, ckpt_best)
            print(f"  -> saved BEST to {ckpt_best} (val_loss={best_val:.5f})")

    print("Training Stage B finito.")
    print(f"- Best ckpt: {ckpt_best}")
    print(f"- Log CSV  : {csv_path}")

if __name__ == "__main__":
    main()

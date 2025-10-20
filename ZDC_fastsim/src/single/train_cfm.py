#!/usr/bin/env python3
# CFM_vanilla_fullmatrix.py — Conditional Flow Matching "vanilla" (1-step) su matrice 44x44 originale
# Modalità:
#   - train : addestra CFM su immagini raw (no centering/no sum-normalization)
#   - infer : genera campioni condizionati (da test manifest o da CSV di condizioni)
#   - eval  : calcola metriche (MSE/MAE/Δsum/Δcentroid) su test; opzionale confronto con 3-stage

from __future__ import annotations
import os, math, time, argparse, json, gzip, csv
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------- Costanti --------------------
H = 44
W = 44

# Default delle 7 variabili richieste: Energia, vertice 3D, impulso 3D
COND_COLS_DEFAULT = ["E","x_vtx","y_vtx","z_vtx","px","py","pz"]

# -------------------- Utils IO --------------------
def load_json_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def append_csv(path: Path, row: Dict[str, Any], header: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not path.exists()) and header is not None:
        with open(path, "w") as f:
            pd.DataFrame(columns=header).to_csv(f, header=True, index=False)
    if row:
        with open(path, "a") as f:
            pd.DataFrame([row]).to_csv(f, header=False, index=False)

# -------------------- Dataset (immagini RAW) --------------------
class FullMatrixCFMDataset(Dataset):
    """
    Carica righe dal compact parquet *e* le immagini (RAW):
      - preferisce LMDB se presente nel manifest
      - fallback a shard .pkl con colonna immagine
    NON applica traslazioni o normalizzazioni: restituisce la matrice così com'è.
    Opzione: --intensity-scale per riscalare le intensità (default 1.0, cioè identità).
    """
    def __init__(self,
                 rows_meta: List[Dict[str, Any]],
                 cond_cols: List[str],
                 intensity_scale: float = 1.0,
                 clip_min: float = 0.0):
        super().__init__()
        self.cond_cols = cond_cols
        self.intensity_scale = float(intensity_scale)
        self.clip_min = float(clip_min)

        # carica tutti i parquet e mappa globale→locale
        self.parquets: List[pd.DataFrame] = []
        self.global_to_local: List[Tuple[int,int]] = []
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
        if arr.shape != (H, W):
            arr = arr.astype(np.float32, copy=False)
            if arr.shape != (H, W):
                raise ValueError(f"Image shape mismatch {shard_src}[{idx}]: {arr.shape} != {(H,W)}")
        return arr.astype(np.float32, copy=False)

    def __getitem__(self, i: int):
        pid, rid = self.global_to_local[i]
        row = self.parquets[pid].iloc[rid]
        io = self.per_pq_io[pid]

        # 1) immagine RAW
        if io["lmdb"]:
            img_np = self._read_image_from_lmdb(io["lmdb"], rid)
        elif io["shard_src"] and io["image_col"]:
            img_np = self._read_image_from_pkl(io["shard_src"], io["image_col"], rid)
        else:
            raise RuntimeError("Impossibile leggere immagine: né LMDB né shard_src disponibili.")

        # evita warning "non-writable" di PyTorch e applica clip/scala opzionale
        img_np = np.array(img_np, dtype=np.float32, copy=True)
        if self.clip_min is not None:
            img_np = np.maximum(img_np, self.clip_min)
        if self.intensity_scale != 1.0:
            img_np = img_np * self.intensity_scale

        x0 = torch.from_numpy(img_np).unsqueeze(0).to(torch.float32)  # [1,H,W]

        # 2) cond features (7 variabili di default)
        cond_vals = [float(row[c]) for c in self.cond_cols]
        c = torch.tensor(cond_vals, dtype=torch.float32)  # [C]

        return x0, c

# -------------------- Modello: U-Net 2D compatto --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(min(groups, out_ch), out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(min(groups, out_ch), out_ch), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class UNetCFM(nn.Module):
    """
    U-Net condizionale con iniezione di condizione+t come canale.
    Nota: architettura semplice ma efficace per CFM "vanilla".
    """
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
        tc = torch.cat([t.view(B,1), c], dim=1)      # [B, 1+C]
        c_embed = self.c_proj(tc).view(B, -1, 1, 1)  # [B, base, 1, 1]
        c_map = c_embed[:, :1].repeat(1, 1, H, W)    # usa 1 canale per semplicità
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

# -------------------- CFM loss --------------------
def cfm_loss(model, x0, c):
    """
    Conditional Flow Matching (dati raw):
      x_t = (1-t) x0 + t x1, con x1 ~ N(0, I)
      dx/dt = x1 - x0
      v_theta ≈ E[ || v_theta(x_t, t, c) - (x1 - x0) ||^2 ]
    """
    B = x0.size(0)
    t = torch.rand(B, device=x0.device).view(B,1,1,1)  # [B,1,1,1]
    x1 = torch.randn_like(x0)
    xt = (1.0 - t) * x0 + t * x1
    target = x1 - x0
    pred = model(xt, t.view(B,1), c)
    return F.mse_loss(pred, target)

# -------------------- Integrazione (sampling) --------------------
@torch.no_grad()
def sample_cfm(model: UNetCFM, c: torch.Tensor, steps: int = 200, shape=(1,1,H,W), device="cuda"):
    """
    Integra dx/dt = v_theta(x,t,c) da t=1 → 0 (Euler esplicito).
    x_1 ~ N(0, I).
    """
    model.eval()
    B = c.size(0)
    x = torch.randn((B, *shape[1:]), device=device)
    t_vals = torch.linspace(1.0, 0.0, steps+1, device=device)
    for i in range(steps):
        t = t_vals[i].expand(B,1)
        v = model(x, t, c)           # dx/dt
        dt = t_vals[i+1] - t_vals[i] # negativo
        x = x + v * dt
    return x  # [B,1,H,W]

# -------------------- Metriche --------------------
def image_metrics(pred: np.ndarray, real: np.ndarray):
    # pred/real: [H,W]
    mse = float(np.mean((pred - real)**2))
    mae = float(np.mean(np.abs(pred - real)))
    sum_pred = float(np.sum(pred))
    sum_real = float(np.sum(real))
    dsum = float(sum_pred - sum_real)

    # centroidi (pesati dalle intensità, gestisce casi degeneri)
    def centroid(img):
        s = img.sum()
        if s <= 0:
            return (np.nan, np.nan)
        ys, xs = np.indices(img.shape)
        cx = float((xs*img).sum()/s)
        cy = float((ys*img).sum()/s)
        return (cx, cy)
    cx_p, cy_p = centroid(pred)
    cx_r, cy_r = centroid(real)
    dcx = float(cx_p - cx_r) if (not np.isnan(cx_p) and not np.isnan(cx_r)) else float("nan")
    dcy = float(cy_p - cy_r) if (not np.isnan(cy_p) and not np.isnan(cy_r)) else float("nan")
    return {"mse": mse, "mae": mae, "dsum": dsum, "dcx": dcx, "dcy": dcy}

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="CFM vanilla (full matrix 44x44, no centering/no normalization)")
    ap.add_argument("--mode", choices=["train","infer","eval"], required=True)

    # dati & IO
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/",
                    help="Cartella con train_compact/, test_compact/, stats/ (e manifest.json.gz)")
    ap.add_argument("--out-dir", default="./runs_cfm_vanilla",
                    help="Dove salvare ckpt, log, samples, metrics")
    ap.add_argument("--ckpt", default="", help="Checkpoint per infer/eval (se vuoto usa out-dir/vanilla_cfm_best.pt)")

    # modello/train
    ap.add_argument("--cond-cols", type=str, default=",".join(COND_COLS_DEFAULT),
                    help="Colonne condizionali (comma-sep)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train-shards", type=int, default=0,
                    help="Usa i primi N shard di train (<=0 per tutti)")

    # dataset trasformazioni (senza cambiare la 'fisica' dell'output)
    ap.add_argument("--intensity-scale", type=float, default=1.0,
                    help="Fattore moltiplicativo applicato alle immagini in train/infer (default 1.0).")
    ap.add_argument("--clip-min", type=float, default=0.0,
                    help="Clippa a questo minimo (default 0.0).")

    # infer
    ap.add_argument("--steps", type=int, default=200, help="Passi di integrazione Euler da t=1→0")
    ap.add_argument("--n-samples", type=int, default=1, help="Campioni stocastici per condizione")
    ap.add_argument("--from-test", action="store_true",
                    help="In infer: genera per l'intero test set del manifest.")
    ap.add_argument("--conds-csv", type=str, default="",
                    help="CSV esterno con colonne cond (stesso ordine di --cond-cols). Se non vuoto, genera per queste righe.")
    ap.add_argument("--out-samples-dir", type=str, default="",
                    help="Cartella per salvare .npy dei campioni; default: out-dir/samples_vanilla/")

    # eval
    ap.add_argument("--samples-dir", type=str, default="",
                    help="Cartella con i .npy generati (default: out-dir/samples_vanilla/)")
    ap.add_argument("--metrics-csv", type=str, default="",
                    help="File CSV per scrivere metriche (default: out-dir/metrics_vanilla.csv)")
    ap.add_argument("--compare-dir", type=str, default="",
                    help="Opzionale: cartella con .npy generati dal modello 3-stage con stessi nomi test_XXXXXX.npy per confronto")

    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    compact_dir = Path(args.compact_dir)
    manifest_path = compact_dir / "stats" / "manifest.json.gz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")

    manifest = load_json_gz(str(manifest_path))
    train_meta_all: List[Dict[str, Any]] = manifest["train"]
    test_meta  = manifest["test"]

    # selezione shard train
    use_n = int(args.train_shards)
    if use_n <= 0 or use_n > len(train_meta_all):
        use_n = len(train_meta_all)
    train_meta = train_meta_all[:use_n]

    # prepara meta per dataset
    rows_train = [{
        "parquet": m["parquet"],
        "lmdb": m.get("lmdb", None),
        "shard_src": m.get("shard_src", None),
        "image_col": m.get("image_col", None),
    } for m in train_meta]
    rows_test = [{
        "parquet": test_meta["parquet"],
        "lmdb": test_meta.get("lmdb", None),
        "shard_src": test_meta.get("shard_src", None),
        "image_col": test_meta.get("image_col", None),
    }]

    cond_cols = [s.strip() for s in args.cond_cols.split(",") if s.strip()]
    cond_dim = len(cond_cols)

    out_dir = Path(args.out-dir if hasattr(args, "out-dir") else args.out_dir)  # safeguard
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt) if args.ckpt else (out_dir / "vanilla_cfm_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        print(f"[INFO] TRAIN  shards={use_n}/{len(train_meta_all)}  batch={args.batch_size}  cond_dim={cond_dim}")
        ds_tr = FullMatrixCFMDataset(rows_train, cond_cols,
                                     intensity_scale=args.intensity_scale,
                                     clip_min=args.clip_min)
        ds_te = FullMatrixCFMDataset(rows_test, cond_cols,
                                     intensity_scale=args.intensity_scale,
                                     clip_min=args.clip_min)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)

        model = UNetCFM(cond_dim=cond_dim, base=args.base_ch).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        csv_path = out_dir / "train_log.csv"
        best_val = float("inf")
        append_csv(csv_path, {}, header=[
            "epoch","train_loss","val_loss","lr","time_sec","batch_size",
            "cond_cols","intensity_scale","clip_min","train_shards_used"
        ])

        for ep in range(1, args.epochs+1):
            t0 = time.time()
            # ---- train ----
            model.train(); run, seen = 0.0, 0
            for it, (x0, c) in enumerate(dl_tr):
                x0 = x0.to(device, non_blocking=True); c = c.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = cfm_loss(model, x0, c)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run += float(loss.item()) * x0.size(0); seen += x0.size(0)
                if args.print_every and (it+1) % args.print_every == 0:
                    print(f"  [ep {ep:03d}] it={it+1:05d} train_loss={run/seen:.6f}")
            train_loss = run / max(1, seen)

            # ---- val ----
            model.eval(); v_run, v_seen = 0.0, 0
            with torch.no_grad():
                for x0, c in dl_te:
                    x0 = x0.to(device); c = c.to(device)
                    v = cfm_loss(model, x0, c)
                    v_run += float(v.item()) * x0.size(0); v_seen += x0.size(0)
            val_loss = v_run / max(1, v_seen)
            dt = time.time() - t0
            print(f"[EP {ep:03d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  ({dt:.1f}s)")

            append_csv(csv_path, {
                "epoch": ep,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(opt.param_groups[0]["lr"]),
                "time_sec": float(dt),
                "batch_size": int(args.batch_size),
                "cond_cols": "|".join(cond_cols),
                "intensity_scale": float(args.intensity_scale),
                "clip_min": float(args.clip_min),
                "train_shards_used": int(use_n),
            })

            if val_loss < best_val and np.isfinite(val_loss):
                best_val = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "cond_cols": cond_cols,
                    "H": H, "W": W,
                    "base_ch": args.base_ch,
                    "intensity_scale": float(args.intensity_scale),
                    "clip_min": float(args.clip_min),
                }, ckpt_path)
                print(f"  -> saved BEST to {ckpt_path} (val_loss={best_val:.6f})")

        print("Training finito.")
        print(f"- Best ckpt: {ckpt_path}")
        print(f"- Log CSV  : {csv_path}")
        return

    # ---- infer / eval: carica modello ----
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = UNetCFM(cond_dim=len(ckpt.get("cond_cols", cond_cols)),
                    base=ckpt.get("base_ch", args.base_ch))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    cond_cols = ckpt.get("cond_cols", cond_cols)
    print(f"[INFO] Caricato ckpt {ckpt_path}  (cond_cols={cond_cols})")

    if args.mode == "infer":
        # prepara condizioni
        samples_dir = Path(args.out_samples_dir) if args.out_samples_dir else (out_dir / "samples_vanilla")
        samples_dir.mkdir(parents=True, exist_ok=True)

        cond_df = None
        indices = None
        if args.from_test:
            # genera per tutto il test set nel suo ordine
            pq = pd.read_parquet(test_meta["parquet"])
            cond_df = pq[cond_cols].copy()
            indices = list(range(len(cond_df)))
            print(f"[INFO] Inferenza su test set: {len(indices)} righe.")
        elif args.conds_csv:
            cond_df = pd.read_csv(args.conds_csv)
            missing = [c for c in cond_cols if c not in cond_df.columns]
            if missing:
                raise ValueError(f"Nel CSV mancano colonne cond: {missing}")
            indices = list(range(len(cond_df)))
            print(f"[INFO] Inferenza da CSV: {len(indices)} righe.")
        else:
            raise ValueError("Usa --from-test oppure fornisci --conds-csv per l'inferenza.")

        # batch infer
        B = 64
        all_c = torch.tensor(cond_df[cond_cols].values, dtype=torch.float32)
        n = all_c.size(0)
        k = int(args.n_samples)

        with torch.no_grad():
            for start in range(0, n, B):
                end = min(start+B, n)
                c_batch = all_c[start:end].to(device)
                for s in range(k):
                    x = sample_cfm(model, c_batch, steps=args.steps, shape=(1,1,H,W), device=device)
                    # inverti qualsiasi intensity_scale applicata in train (qui è moltiplicativa)
                    scale_inv = float(ckpt.get("intensity_scale", 1.0))
                    if scale_inv != 1.0:
                        x = x / scale_inv
                    x_np = x.squeeze(1).cpu().numpy()  # [b,H,W]
                    for i, arr in enumerate(x_np):
                        idx = start + i
                        name = f"test_{idx:06d}" if args.from_test else f"row_{idx:06d}"
                        suffix = f"_s{s:02d}" if k > 1 else ""
                        np.save(samples_dir / f"{name}{suffix}.npy", arr.astype(np.float32, copy=False))
                print(f"  [infer] {end}/{n} righe generate...")

        print(f"Inferenza completata. Output: {samples_dir}")
        return

    if args.mode == "eval":
        # prepara dataset test per ground truth
        pq = pd.read_parquet(test_meta["parquet"])
        # per coerenza nell'ordine coi file generati assumiamo indice di riga = id file
        n = len(pq)
        samples_dir = Path(args.samples_dir) if args.samples_dir else (out_dir / "samples_vanilla")
        if not samples_dir.exists():
            raise FileNotFoundError(f"samples-dir non esiste: {samples_dir}")

        metrics_csv = Path(args.metrics_csv) if args.metrics_csv else (out_dir / "metrics_vanilla.csv")
        header = ["index","mse","mae","dsum","dcx","dcy"]
        if args.compare_dir:
            header += ["mse_cmp","mae_cmp","dsum_cmp","dcx_cmp","dcy_cmp"]
        append_csv(metrics_csv, {}, header=header)

        # per leggere immagini reali dal test shard
        # se c'è lmdb lo usiamo, altrimenti shard_src
        io = {
            "lmdb": test_meta.get("lmdb", None),
            "shard_src": test_meta.get("shard_src", None),
            "image_col": test_meta.get("image_col", None),
        }
        pkl_cache = {}
        def read_real(idx: int) -> np.ndarray:
            if io["lmdb"]:
                import lmdb
                env = lmdb.open(io["lmdb"], readonly=True, lock=False)
                with env.begin() as txn:
                    key = f"{idx:08d}".encode("ascii")
                    buf = txn.get(key)
                    if buf is None:
                        env.close()
                        raise KeyError(f"Idx {idx} non trovato in LMDB {io['lmdb']}")
                    arr = np.frombuffer(buf, dtype=np.float32).reshape(H, W)
                env.close()
                return arr
            elif io["shard_src"] and io["image_col"]:
                if io["shard_src"] not in pkl_cache:
                    pkl_cache[io["shard_src"]] = pd.read_pickle(io["shard_src"])
                df = pkl_cache[io["shard_src"]]
                val = df.iloc[idx][io["image_col"]]
                arr = np.asarray(val)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                return arr.astype(np.float32, copy=False)
            else:
                raise RuntimeError("Impossibile leggere immagine reale (né LMDB né shard_src).")

        compare_dir = Path(args.compare_dir) if args.compare_dir else None

        for idx in range(n):
            pred_path = samples_dir / f"test_{idx:06d}.npy"
            if not pred_path.exists():
                # prova con suffisso s00 (nel caso n-samples>1)
                pred_candidates = sorted(samples_dir.glob(f"test_{idx:06d}_s*.npy"))
                if not pred_candidates:
                    continue
                pred_path = pred_candidates[0]
            pred = np.load(pred_path)
            real = read_real(idx)
            met = image_metrics(pred, real)

            row = {"index": idx,
                   "mse": met["mse"], "mae": met["mae"], "dsum": met["dsum"],
                   "dcx": met["dcx"], "dcy": met["dcy"]}

            if compare_dir:
                pred2_path = compare_dir / f"test_{idx:06d}.npy"
                if pred2_path.exists():
                    pred2 = np.load(pred2_path)
                else:
                    cand2 = sorted(compare_dir.glob(f"test_{idx:06d}_s*.npy"))
                    pred2 = np.load(cand2[0]) if cand2 else None
                if pred2 is not None:
                    met2 = image_metrics(pred2, real)
                    row.update({
                        "mse_cmp": met2["mse"], "mae_cmp": met2["mae"], "dsum_cmp": met2["dsum"],
                        "dcx_cmp": met2["dcx"], "dcy_cmp": met2["dcy"],
                    })

            append_csv(metrics_csv, row)

            if (idx+1) % 500 == 0:
                print(f"  [eval] {idx+1}/{n} righe valutate...")

        print(f"Valutazione completata. Metriche: {metrics_csv}")
        return

if __name__ == "__main__":
    main()

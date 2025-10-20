#!/usr/bin/env python3
# ============================================================
# Stage B — Conditional DDPM (shape 44x44 centrata e normalizzata)
# - Dati: come CFM (LMDB se presente, fallback shard .pkl) + manifest json.gz
# - Target: shape [1,44,44] centrata, somma=1 (il Dataset la fornisce già così)
# - Modello: UNet 2D compatto con timestep embedding + cond embedding
# - Loss: MSE( eps_pred, eps )
# - Val: stessa loss su split test
# - Opzionali: AMP (--amp), EMA (--ema)
# - Checkpoint: best su val_loss con meta (cond_cols, H, W, betas)
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
from torch.utils.data import Dataset, DataLoader

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

# -------------------- Dataset --------------------
class ShapeDDPMDataset(Dataset):
    """
    Identico al tuo ShapeCFMDataset: carica immagine, centra via (x_imp,y_imp) o da immagine,
    normalizza a somma=1, e restituisce x0:[1,H,W], c:[C].
    """
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
        if arr.shape != (H, W):
            arr = arr.astype(np.float32, copy=False)
            if arr.shape != (H, W):
                raise ValueError(f"Image shape mismatch {arr.shape} != {(H,W)}")
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
        # centra + normalizza
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
        c = torch.tensor(cond_vals, dtype=torch.float32)  # [C]
        return x0, c

# -------------------- Sinusoidal timestep embedding --------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        # t in [0, T-1] (int) oppure [0,1] (float). Qui assumiamo [0,1] float e riscalamo.
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1))
        )
        # usa t in [0,1]
        args = t.float() * freqs  # [B,half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # [B, dim]

# -------------------- UNet per DDPM (ε-pred) --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class UNetDDPM(nn.Module):
    def __init__(self, cond_dim: int, base: int = 32, tdim: int = 64):
        super().__init__()
        # Embedding tempo + cond → FiLM / canale extra
        self.t_emb = SinusoidalPosEmb(tdim)
        self.t_mlp = nn.Sequential(nn.Linear(tdim, base), nn.SiLU(), nn.Linear(base, base))
        self.c_proj = nn.Linear(cond_dim, base)

        self.enc1 = DoubleConv(1+1, base)   # canale extra per (t_emb + c_emb) fuso -> 1 canale
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
        """
        x: [B,1,H,W] = x_t
        t01: [B,1]   = tempo normalizzato in [0,1], 1 = molto rumoroso
        c: [B,C]
        Ritorna: eps_pred stessa shape di x
        """
        B = x.size(0)
        t_emb = self.t_emb(t01.view(B,1))          # [B,tdim]
        t_emb = self.t_mlp(t_emb)                  # [B,base]
        c_emb = self.c_proj(c)                     # [B,base]
        fuse = (t_emb + c_emb).view(B,-1,1,1).repeat(1,1,H,W)
        xc = torch.cat([x, fuse[:, :1]], dim=1)    # 1 canale per mantenere arch minima

        h1 = self.enc1(xc)
        h2 = self.enc2(self.down1(h1))
        h3 = self.mid(self.down2(h2))
        u2 = self.up2(h3)
        d2 = self.dec2(torch.cat([u2, h2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, h1], dim=1))
        eps = self.out(d1)
        return eps

# -------------------- Schedules DDPM --------------------
def betas_linear(T: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def betas_cosine(T: int, s: float=0.008):
    # da Nichol & Dhariwal 2021
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-8, 0.999)

# q(x_t | x_0)
def q_sample(x0: torch.Tensor, t_idx: torch.Tensor, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    x0: [B,1,H,W], t_idx: [B] int64 in [0..T-1]
    """
    B = x0.size(0)
    noise = torch.randn_like(x0)
    fac1 = sqrt_alphas_cumprod[t_idx].view(B,1,1,1)
    fac2 = sqrt_one_minus_alphas_cumprod[t_idx].view(B,1,1,1)
    x_t = fac1 * x0 + fac2 * noise
    return x_t, noise

# -------------------- EMA semplice --------------------
class EMA:
    def __init__(self, model: nn.Module, mu: float=0.9999):
        self.mu = mu
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.mu).add_(v.detach(), alpha=1.0 - self.mu)
    def copy_to(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(self.shadow[k])

# -------------------- CSV logger --------------------
def append_csv(path: Path, row: Dict[str, Any], header: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not path.exists()) and header is not None:
        with open(path, "w") as f:
            pd.DataFrame(columns=header).to_csv(f, header=True, index=False)
    if row:
        with open(path, "a") as f:
            pd.DataFrame([row]).to_csv(f, header=False, index=False)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Stage B — Conditional DDPM (shape 44x44)")
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/",
                    help="Cartella con train_compact/, test_compact/, stats/ (e manifest.json.gz)")
    ap.add_argument("--out-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_B_DDPM",
                    help="Dove salvare ckpt e CSV")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--cond-cols", type=str, default=",".join(COND_COLS_DEFAULT))
    ap.add_argument("--recompute-from-image", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--print-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train-shards", type=int, default=1)

    # DDPM specific
    ap.add_argument("--timesteps", type=int, default=1000, help="Numero passi di rumore T")
    ap.add_argument("--betas", choices=["linear","cosine"], default="cosine")
    ap.add_argument("--ema", action="store_true", help="Usa EMA dei pesi (consigliato)")
    ap.add_argument("--amp", action="store_true", help="Mixed precision in training")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    compact_dir = Path(args.compact_dir)
    manifest_path = compact_dir / "stats" / "manifest.json.gz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}. Genera i compact con dataset_builder.py")

    manifest = load_json_gz(str(manifest_path))
    train_meta_all: List[Dict[str, Any]] = manifest["train"]
    test_meta  = manifest["test"]

    use_n = int(args.train_shards) if args.train_shards is not None else 1
    if use_n <= 0 or use_n > len(train_meta_all):
        use_n = len(train_meta_all)
    train_meta = train_meta_all[:use_n]
    print(f"[INFO] Userò {use_n} shard di train su {len(train_meta_all)} totali.")

    rows_train = [ {"parquet": m["parquet"], "lmdb": m.get("lmdb"), "shard_src": m.get("shard_src"), "image_col": m.get("image_col")} for m in train_meta ]
    rows_test  = [ {"parquet": test_meta["parquet"], "lmdb": test_meta.get("lmdb"), "shard_src": test_meta.get("shard_src"), "image_col": test_meta.get("image_col")} ]

    cond_cols = [s.strip() for s in args.cond_cols.split(",") if s.strip()]
    cond_dim = len(cond_cols)
    print(f"[INFO] Start Stage B DDPM: epochs={args.epochs}  batch={args.batch_size}  cond_dim={cond_dim}  recompute_from_image={args.recompute_from_image}")

    ds_tr = ShapeDDPMDataset(rows_train, cond_cols, recompute_from_image=args.recompute_from_image)
    ds_te = ShapeDDPMDataset(rows_test,  cond_cols, recompute_from_image=args.recompute_from_image)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0,                pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDDPM(cond_dim=cond_dim, base=32, tdim=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # schedule
    T = int(args.timesteps)
    if args.betas == "linear":
        betas = betas_linear(T)
    else:
        betas = betas_cosine(T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = out_dir / "stageB_ddpm_best.pt"
    csv_path  = out_dir / "stageB_ddpm_train_log.csv"

    best_val = float("inf")
    header = ["epoch","train_loss","val_loss","lr","time_sec","batch_size","cond_cols","recompute_from_image","train_shards_used","timesteps","betas","ema","amp"]
    append_csv(csv_path, {}, header=header)

    # EMA opzionale
    ema = EMA(model, mu=0.9999) if args.ema else None

    def ddpm_loss_batch(x0, c):
        """
        - Estrae t ~ Uniform{0..T-1}
        - Costruisce x_t = sqrt(ᾱ_t) x0 + sqrt(1-ᾱ_t) ε
        - Predice ε̂(x_t, t, c), loss = MSE(ε̂, ε)
        """
        B = x0.size(0)
        t_idx = torch.randint(0, T, (B,), device=x0.device, dtype=torch.long)
        x_t, noise = q_sample(x0, t_idx, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        # normalizza t in [0,1] (1=più rumoroso)
        t01 = (t_idx.float() + 1e-8) / float(T-1)
        eps_pred = model(x_t, t01.view(B,1), c)
        return F.mse_loss(eps_pred, noise)

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        # ---- train ----
        model.train()
        run, seen = 0.0, 0
        for it, (x0, c) in enumerate(dl_tr):
            x0 = x0.to(device, non_blocking=True)
            c  = c.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = ddpm_loss_batch(x0, c)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema.update(model)
            run += float(loss.item()) * x0.size(0); seen += x0.size(0)
            if args.print_every and (it+1) % args.print_every == 0:
                print(f"  [ep {ep:03d}] it={it+1:05d} train_loss={run/seen:.5f}")
        train_loss = run / max(1,seen)

        # ---- val ---- (usa i pesi EMA se disponibili)
        model.eval()
        if ema is not None:
            _backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
        v_run, v_seen = 0.0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            for x0, c in dl_te:
                x0 = x0.to(device); c = c.to(device)
                v = ddpm_loss_batch(x0, c)
                v_run += float(v.item()) * x0.size(0); v_seen += x0.size(0)
        if ema is not None:
            # ripristina pesi normali
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    v.copy_(_backup[k])
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
            "recompute_from_image": int(args.recompute_from_image),
            "train_shards_used": int(use_n),
            "timesteps": int(T),
            "betas": str(args.betas),
            "ema": int(args.ema),
            "amp": int(args.amp),
        })

        if val_loss < best_val and np.isfinite(val_loss):
            best_val = val_loss
            save_obj = {
                "model": model.state_dict(),
                "cond_cols": cond_cols,
                "H": H, "W": W,
                "timesteps": int(T),
                "betas_kind": args.betas,
                "betas": betas.cpu().numpy().tolist(),  # salva schedule per l'inferenza
                "ema": args.ema,
            }
            if ema is not None:
                # salva anche snapshot EMA
                save_obj["model_ema"] = ema.shadow
            torch.save(save_obj, ckpt_best)
            print(f"  -> saved BEST to {ckpt_best} (val_loss={best_val:.6f})")

    print("Training DDPM Stage B finito.")
    print(f"- Best ckpt: {ckpt_best}")
    print(f"- Log CSV  : {csv_path}")

if __name__ == "__main__":
    main()

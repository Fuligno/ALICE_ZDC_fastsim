#!/usr/bin/env python3
# Stage C — Conditional Flow Matching (CFM) per Gamma_tot (1D) condizionato su 7 feature
# - Spazio target: y = standardize(log1p(T + U)), con mean/std da stats.json.gz
# - Base: x0 ~ N(0,1)
# - Interpolazione lineare: x_t = (1 - t) * x0 + t * y1
# - Target velocity: u_t = y1 - x0 (costante in t)
# - Rete: MLP che prende [x_t, t, t^2, t^3, cond7_norm] e predice v_theta(x_t, t, cond)
# - Loss: MSE(v_theta, u_t)
# - Val: stessa loss su test (nuovi campioni x0, t)
# - Nessun log-det/nll: training molto più stabile dei NF
from __future__ import annotations
import os, math, time, argparse, gzip, json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# -------------------------------------------------
# Colonne e alias
# -------------------------------------------------
FEATURE_NAMES = ["E","vx","vy","vz","px","py","pz"]
GAMMA_ALIASES = ["Gamma_tot","T","psum","photonSum"]

# -------------------------------------------------
# IO utils
# -------------------------------------------------
def list_train_parquet(compact_dir: Path) -> List[Path]:
    d = compact_dir / "train_compact"
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nessun parquet trovato in {d}")
    return files

def get_test_parquet(compact_dir: Path) -> Path:
    p = compact_dir / "test_compact" / "test.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Test parquet non trovato: {p}")
    return p

def load_stats(compact_dir: Path) -> Dict[str, Any]:
    sp = compact_dir / "stats" / "stats.json.gz"
    if not sp.exists():
        raise FileNotFoundError(f"Stats non trovate: {sp}")
    with gzip.open(sp, "rt", encoding="utf-8") as f:
        st = json.load(f)
    for k in ("mean7","std7","logT_mean","logT_std"):
        if k not in st:
            raise KeyError(f"Chiave {k} assente in stats.")
    return st

def find_gamma_col(df: pd.DataFrame) -> str:
    for c in GAMMA_ALIASES:
        if c in df.columns:
            return c
    raise KeyError(f"Nessuna colonna Gamma_tot trovata. Cerco alias: {GAMMA_ALIASES}")

# -------------------------------------------------
# Dataset iterabile: (cond7_norm[7], y1_standardized[scalar])
# -------------------------------------------------
class GammaCFMIterable(IterableDataset):
    def __init__(self, parquet_paths: List[Path], mean7: np.ndarray, std7: np.ndarray,
                 logT_mean: float, logT_std: float,
                 shuffle_buffer: int = 40000, seed: int = 123):
        super().__init__()
        self.paths = [Path(p) for p in parquet_paths]
        self.mean7 = mean7.astype(np.float32)
        self.std7  = std7.astype(np.float32)
        self.logT_mean = float(logT_mean)
        self.logT_std  = float(logT_std)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)

    def _iter_worker(self, paths_subset: List[Path]):
        rng = np.random.RandomState(self.seed + (get_worker_info().id if get_worker_info() else 0))
        buf = []
        for p in paths_subset:
            df = pd.read_parquet(p)
            gcol = find_gamma_col(df)
            X = df[FEATURE_NAMES].to_numpy(np.float32, copy=False)   # [N,7]
            T = df[gcol].to_numpy(np.float32, copy=False)            # [N]
            Xn = (X - self.mean7[None,:]) / (self.std7[None,:] + 1e-6)
            # y = standardize(log1p(T + U))
            U = rng.rand(T.shape[0]).astype(np.float32)
            y = np.log1p(T + U).astype(np.float32)
            y1 = (y - self.logT_mean) / (self.logT_std + 1e-6)

            for i in range(Xn.shape[0]):
                buf.append( (torch.from_numpy(Xn[i]), float(y1[i])) )
                if len(buf) >= self.shuffle_buffer:
                    j = rng.randint(0, len(buf))
                    yield buf.pop(j)

            rng.shuffle(buf)
            while buf:
                yield buf.pop()

            del df, X, T, Xn, U, y, y1

    def __iter__(self):
        info = get_worker_info()
        paths = list(self.paths)
        import random as _r
        _r.Random(self.seed + (info.id if info else 0)).shuffle(paths)
        if info is None:
            return self._iter_worker(paths)
        subset = [paths[i] for i in range(len(paths)) if i % info.num_workers == info.id]
        return self._iter_worker(subset)

# -------------------------------------------------
# Modello: MLP per v_theta(x_t, t, cond7)
# -------------------------------------------------
class CFMPredictor(nn.Module):
    """
    Input: [x_t (1), t, t^2, t^3, cond7_norm (7)] -> hidden MLP -> v (1)
    """
    def __init__(self, cond_dim=7, hidden=128, layers=3, dropout=0.0):
        super().__init__()
        in_dim = 1 + 3 + cond_dim  # x_t + [t, t^2, t^3] + cond7
        blocks = []
        d = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout > 0: blocks += [nn.Dropout(dropout)]
            d = hidden
        blocks += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x_t, t: (B,), cond: (B,7)
        t2 = t * t
        t3 = t2 * t
        inp = torch.cat([x_t.unsqueeze(1),
                         t.unsqueeze(1), t2.unsqueeze(1), t3.unsqueeze(1),
                         cond], dim=1)
        v = self.net(inp).squeeze(1)  # (B,)
        return v

# -------------------------------------------------
# CLI
# -------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Stage C: Conditional Flow Matching per Gamma_tot")
    ap.add_argument("--compact-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/data/",
                    help="Cartella con train_compact/, test_compact/, stats/")
    ap.add_argument("--out-dir", default="/data/dataalice/dfuligno/ZDC_fastsim/configs/checkpoints_C",
                    help="Dove salvare checkpoint e CSV")
    ap.add_argument("--train-shards", type=int, default=1, help="Quanti shard parquet usare (default 1)")

    # training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--shuffle-buffer", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument("--steps-per-epoch", type=int, default=1000,
                    help="Numero di batch per epoca (IterableDataset non ha len).")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # modello
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    return ap.parse_args()

# -------------------------------------------------
# Main training
# -------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compact_dir = Path(args.compact_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Stats e file
    stats = load_stats(compact_dir)
    mean7 = np.asarray(stats["mean7"], dtype=np.float32)
    std7  = np.asarray(stats["std7"], dtype=np.float32)
    logT_mean = float(stats["logT_mean"])
    logT_std  = float(stats["logT_std"])

    train_all = list_train_parquet(compact_dir)
    train_files = train_all[:max(1, min(args.train_shards, len(train_all)))]
    test_file = get_test_parquet(compact_dir)
    print(f"[DATA] Train shards: {len(train_files)} / {len(train_all)}")

    # Dataset/Dataloader
    train_ds = GammaCFMIterable(train_files, mean7, std7, logT_mean, logT_std,
                                shuffle_buffer=args.shuffle_buffer, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers>0 else None,
        persistent_workers=(args.num_workers>0),
        drop_last=True
    )

    # Test tensors (fisso cond & y1 per valutazione loss)
    df_te = pd.read_parquet(test_file)
    gamma_col = find_gamma_col(df_te)
    X_te = df_te[FEATURE_NAMES].to_numpy(np.float32, copy=False)
    T_te = df_te[gamma_col].to_numpy(np.float32, copy=False)
    X_te_n = (X_te - mean7[None,:]) / (std7[None,:] + 1e-6)
    # dequantization per valutazione (come train)
    U_te = np.random.rand(T_te.shape[0]).astype(np.float32)
    y_te = np.log1p(T_te + U_te).astype(np.float32)
    y1_te = (y_te - logT_mean) / (logT_std + 1e-6)
    X_te_t = torch.from_numpy(X_te_n)
    y1_te_t = torch.from_numpy(y1_te)

    # Modello
    model = CFMPredictor(cond_dim=7, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(torch.cuda.is_available() and not args.no_amp))

    # Scheduler (warmup + cosine sugli step)
    total_steps  = args.epochs * max(1, args.steps_per_epoch)
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    class WarmupCosine:
        def __init__(self, optimizer, warmup, total, base_lr):
            self.opt=optimizer; self.warm=max(1,warmup); self.total=max(total,self.warm+1)
            self.base=base_lr; self.t=0
        def step(self):
            self.t+=1
            if self.t<=self.warm:
                lr=self.base*self.t/self.warm
            else:
                prog=(self.t-self.warm)/(self.total-self.warm)
                lr=0.15*self.base+0.85*self.base*0.5*(1+math.cos(math.pi*prog))
            for pg in self.opt.param_groups: pg['lr']=lr
            return lr
    sched = WarmupCosine(opt, warmup_steps, total_steps, args.lr)

    # CSV log
    metrics_csv = out_dir / "stageC_gammaCFM_metrics.csv"
    if not metrics_csv.exists():
        pd.DataFrame([{
            "epoch": 0, "train_fm": np.nan, "test_fm": np.nan,
            "lr": args.lr, "batch_size": args.batch_size,
            "train_shards": len(train_files), "epoch_time_sec": np.nan
        }]).to_csv(metrics_csv, index=False)

    best_val = float("inf")
    best_path = out_dir / "gammaCFM_best.pt"
    print(f"[INFO] Start training C (Clean CFM): epochs={args.epochs} batch={args.batch_size} lr={args.lr}")

    global_step = 0
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        run_loss = 0.0; seen = 0; step = 0

        for cond_cpu, y1_cpu in train_loader:
            B = cond_cpu.size(0)
            cond = cond_cpu.to(device, non_blocking=True).float()     # [B,7]
            y1   = y1_cpu.to(device, non_blocking=True).float()       # [B]

            # campiona x0 ~ N(0,1), t ~ U(0,1), definisci x_t e target velocity
            x0 = torch.randn(B, device=device)
            t  = torch.rand(B, device=device)
            x_t = (1.0 - t) * x0 + t * y1
            u_t = (y1 - x0)  # (B,) — costante in t

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and not args.no_amp)):
                v = model(x_t, t, cond)
                loss = F.mse_loss(v, u_t, reduction="mean")

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            sched.step(); global_step += 1
            run_loss += float(loss.detach()) * B; seen += B; step += 1

            if args.print_every and (step % args.print_every == 0):
                print(f"  [ep {ep:03d}] step {step:05d} | FM_loss {run_loss/seen:.5f}")

            if step >= args.steps_per_epoch:
                break

        train_fm = run_loss / max(1, seen)

        # ---- Val ----
        model.eval()
        with torch.no_grad():
            Xt = X_te_t.to(device)
            y1t= y1_te_t.to(device)
            BS = 16384
            vals = []
            for i in range(0, Xt.size(0), BS):
                cond = Xt[i:i+BS]
                y1   = y1t[i:i+BS]
                B = cond.size(0)
                x0 = torch.randn(B, device=device)
                t  = torch.rand(B, device=device)
                x_t = (1.0 - t) * x0 + t * y1
                u_t = (y1 - x0)
                v = model(x_t, t, cond)
                vals.append(F.mse_loss(v, u_t, reduction="mean").item())
            val_fm = float(np.mean(vals)) if vals else float("nan")

        dt = time.time() - t0
        print(f"[EP {ep:03d}] train_fm={train_fm:.4f} | test_fm={val_fm:.4f} | {dt:.1f}s")

        # CSV append
        pd.DataFrame([{
            "epoch": ep, "train_fm": float(train_fm), "test_fm": float(val_fm),
            "lr": float(opt.param_groups[0]["lr"]), "batch_size": int(args.batch_size),
            "train_shards": int(len(train_files)), "epoch_time_sec": float(dt),
        }]).to_csv(metrics_csv, mode="a", header=False, index=False)

        if val_fm < best_val and np.isfinite(val_fm):
            best_val = val_fm
            payload = {
                "state_dict": model.state_dict(),
                "config": {
                    "cond_dim": 7, "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
                    "mean7": stats["mean7"], "std7": stats["std7"],
                    "logT_mean": float(logT_mean), "logT_std": float(logT_std),
                    "t_feat": ["t","t2","t3"],
                },
                "metrics": {"test_fm": float(val_fm)},
                "epoch": ep,
            }
            torch.save(payload, best_path)
            print(f"  -> saved BEST to {best_path} (test_fm={best_val:.5f})")

    print("Training CFM finito.")
    print(f"Best ckpt in: {out_dir}")
    print(f"CSV:          {metrics_csv}")

if __name__ == "__main__":
    main()

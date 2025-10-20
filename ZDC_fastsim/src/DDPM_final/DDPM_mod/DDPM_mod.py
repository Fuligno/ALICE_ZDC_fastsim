# ============================================================
# DDPM_stage_sharded.py — Two-stage physics-aware diffusion (44x44, conteggi interi)
# - Train su shard .pkl/.plk (uint16 per le immagini) + test.pkl
# - IterableDataset con shuffle shard + buffer, pinned memory, prefetch
# - Statistiche (cond7 mean/std, psum mean/std, y_clip_max) calcolate in streaming
# - Selezione shard via CLI: --train-shards N
# ============================================================
import os, math, gc, time, argparse, random, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# -----------------------
# CONFIG / HYPERPARAMS (default; override da CLI dove previsto)
# -----------------------
DATA_DIR     = "/data/dataalice/dfuligno/tesi/myTesi/Dati_processed"  ### NEW
SEED         = 42
IMG_SIZE     = 44
BATCH_SIZE   = 64
EPOCHS       = 100
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EMA_DECAY    = 0.999
T_STEPS      = 1000

# Eval (snella per epoca)
EVAL_STEPS   = 50
EVAL_NUM     = 16   # numero esempi per split

# Modello
COND7_DIM    = 7
AUX_DIM      = 3                 # (psum_pred_n, x_u, y_u)
COND_DIM     = COND7_DIM + AUX_DIM
BASE_CH      = 64
TIME_DIM     = 64
COND_EMB     = 64
GRAD_CLIP    = 1.0

# CFG
CFG_P_UNCOND   = 0.10
CFG_SCALE_EVAL = 3.0

# P2 weighting
P2_K     = 1.0
P2_GAMMA = 0.5

# Sparsity-aware (in spazio reale)
SPARSE_THRESH  = 1.0
SPARSE_LAMBDA  = 10.0

# Regressione stadio A
LAMBDA_REG_PSUM = 0
LAMBDA_REG_XY   = 0.5

# Physics loss (su x0_hat)
LAMBDA_SUM = 0
LAMBDA_PSUM_MSE      = 0.001   # MSE sulla somma (photon sum)
LAMBDA_CENTER_ARGMAX = 0   # MSE sul "centro" tipo argmax (soft-argmax differenziabile)
LAMBDA_GMAX          = 1e-5   # MSE sul valore massimo (gamma_max)
LAMBDA_ENTROPY       = 0.05   # MSE sull'entropia normalizzata
CENTER_TAU           = 0.02   # temperatura soft-argmax (più bassa = più vicina all'argmax)
ENTROPY_EPS          = 1e-8  # eps numerico per l'entropia

# Target log scaling
Y_CLIP_MAX_DEFAULT = 1024.0

# Prior gaussiano
PRIOR_SIGMA_PX = 2.5

# Anti-NaN / stabilità
MM1_CLAMP_TRAIN = 1.0
MM1_CLAMP_EVAL  = 1.0
NAN_REPLACE_VAL = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except: pass

# FP16 autocast su GPU
AMP_DTYPE = torch.float16

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser(description="Two-stage physics-aware DDPM (sharded)")
parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Cartella con train_shard_*.pkl e test.pkl")
parser.add_argument("--train-shards", type=int, default=None,
                    help="Usa solo i primi N shard (dopo shuffle per epoca). Default: tutti")
parser.add_argument("--shuffle-buffer", type=int, default=10000,
                    help="Buffer per shuffle (approx. shuffle globale dentro shard)")
parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
parser.add_argument("--prefetch-factor", type=int, default=3, help="DataLoader prefetch_factor")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--lr", type=float, default=LR)
parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
parser.add_argument("--use_best", action="store_true",
                    help="Carica EMA UNet + predictor da best_ema_checkpoint.pt")
parser.add_argument("--teacher_p", type=float, default=1.0,
                    help="Probabilità di usare GT (psum/XY) per condizionare B durante il training")
parser.add_argument("--no_compile", action="store_true", help="Disabilita torch.compile (debug)")
parser.add_argument("--no_amp", action="store_true", help="Disabilita autocast/GradScaler (fp32)")
parser.add_argument("--int_eval", action="store_true",
                    help="In eval calcola metriche su immagini INTEGERIZZATE preservando photonSum")
args = parser.parse_args()

BATCH_SIZE   = args.batch_size
EPOCHS       = args.epochs
LR           = args.lr
WEIGHT_DECAY = args.weight_decay
DATA_DIR     = args.data_dir

def unwrap(m): return getattr(m, "_orig_mod", m)

# -----------------------
# SCHEDULE
# -----------------------
def cosine_alpha_bars(T: int, s: float = 0.008, device=DEVICE, dtype=torch.float32):
    steps = torch.arange(T + 1, dtype=dtype, device=device)
    t = steps / T
    a_bars_full = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a_bars_full = a_bars_full / a_bars_full[0]
    alpha_bars = a_bars_full[1:]
    alphas = alpha_bars / a_bars_full[:-1]
    betas = 1.0 - alphas
    return betas.clamp(1e-8, 0.999), alphas, alpha_bars

betas, alphas, alpha_bars = cosine_alpha_bars(T_STEPS, device=DEVICE, dtype=torch.float32)

def gather_ab(t_idx: torch.LongTensor):
    ab = alpha_bars.gather(0, t_idx).view(-1,1,1,1)
    return torch.sqrt(ab), torch.sqrt(1.0 - ab), ab

# -----------------------
# MODELLI
# -----------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Linear(dim*2, dim))
        self.dim = dim
    @staticmethod
    def sinusoidal(t: torch.LongTensor, dim: int, max_period: float = 10000.0):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half-1,1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb
    def forward(self, t): return self.mlp(self.sinusoidal(t, self.dim))

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, groups=8):
        super().__init__()
        g1, g2 = min(groups,in_ch), min(groups,out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch); self.act1 = nn.SiLU(); self.conv1 = nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.norm2 = nn.GroupNorm(g2, out_ch); self.act2 = nn.SiLU(); self.conv2 = nn.Conv2d(out_ch,out_ch,3,padding=1)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch*2))
        self.skip = (in_ch != out_ch);  self.conv_skip = nn.Conv2d(in_ch, out_ch, 1) if self.skip else None
    def forward(self, x, emb):
        scale, shift = self.emb(emb).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1); shift = shift.unsqueeze(-1).unsqueeze(-1)
        one = torch.ones(1, device=x.device, dtype=x.dtype)
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.norm2(h); h = h * (one + scale.to(h.dtype)) + shift.to(h.dtype)
        h = self.conv2(self.act2(h))
        if self.skip: x = self.conv_skip(x)
        return x + h

class Downsample(nn.Module):
    def __init__(self, ch): super().__init__(); self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch): super().__init__(); self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x): return self.conv(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))

def xy_grid(H, W, device, dtype):
    ys = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return gx[None,None], gy[None,None]

def gaussian_prior(x_unit, y_unit, H, W, sigma_px, device, dtype):
    cx = x_unit.to(dtype) * (W - 1)
    cy = y_unit.to(dtype) * (H - 1)
    ys = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    sigma = torch.tensor(sigma_px, device=device, dtype=dtype)
    dy2 = (ys - cy.view(-1,1,1)) ** 2
    dx2 = (xs - cx.view(-1,1,1)) ** 2
    g = torch.exp(-(dx2 + dy2) / (2.0 * sigma ** 2))
    return g.unsqueeze(1)

def gamma_sum_torch(x_real: torch.Tensor) -> torch.Tensor:
    # x_real: [B,1,H,W] >= 0
    return x_real.flatten(1).sum(dim=1)

def gamma_max_torch(x_real: torch.Tensor) -> torch.Tensor:
    return x_real.flatten(1).max(dim=1).values

def entropy_normalized_torch(x_real: torch.Tensor, eps: float = ENTROPY_EPS) -> torch.Tensor:
    B,_,H,W = x_real.shape
    flat = x_real.flatten(1)
    s = flat.sum(dim=1, keepdim=True)
    p = flat / (s + flat.new_tensor(eps))
    h = -(p * (p + p.new_tensor(eps)).log()).sum(dim=1)
    return h / math.log(H*W)

def soft_argmax_xy_unit(x_real: torch.Tensor, tau: float = CENTER_TAU):
    """
    'Centro' approssimazione dell'argmax, ma differenziabile.
    Ritorna coordinate in [0,1]x[0,1].
    """
    B,_,H,W = x_real.shape
    flat = x_real.flatten(1)
    p = torch.softmax(flat / x_real.new_tensor(tau), dim=1)  # approx dell'argmax

    ys = torch.linspace(0, 1, steps=H, device=x_real.device, dtype=x_real.dtype)
    xs = torch.linspace(0, 1, steps=W, device=x_real.device, dtype=x_real.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    gx = grid_x.reshape(1, -1); gy = grid_y.reshape(1, -1)

    cx = (p * gx).sum(dim=1)
    cy = (p * gy).sum(dim=1)
    return cx, cy


class CondUNet(nn.Module):
    def __init__(self, cond_dim=10, time_dim=128, cond_emb=128, base_ch=96):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim,cond_emb), nn.SiLU(), nn.Linear(cond_emb,cond_emb))
        emb_dim = time_dim + cond_emb
        ch = base_ch
        self.stem = nn.Conv2d(4, ch, 3, padding=1)
        self.down1 = ResBlock(ch, ch, emb_dim); self.ds1 = Downsample(ch)
        self.down2 = ResBlock(ch, ch*2, emb_dim); self.ds2 = Downsample(ch*2)
        self.mid1  = ResBlock(ch*2, ch*4, emb_dim); self.mid2 = ResBlock(ch*4, ch*4, emb_dim)
        self.up2   = Upsample(ch*4); self.upb2 = ResBlock(ch*4 + ch*2, ch*2, emb_dim)
        self.up1   = Upsample(ch*2); self.upb1 = ResBlock(ch*2 + ch, ch, emb_dim)
        self.out   = nn.Sequential(nn.GroupNorm(min(8,ch), ch), nn.SiLU(), nn.Conv2d(ch,1,3,padding=1))
    def forward(self, x_t, t, cond_vec, prior_map):
        B,_,H,W = x_t.shape
        gx, gy = xy_grid(H, W, x_t.device, x_t.dtype)
        x_in = torch.cat([x_t, gx.expand(B,1,H,W), gy.expand(B,1,H,W), prior_map.to(x_t.dtype)], dim=1)
        temb = self.time_emb(t)
        cemb = self.cond_mlp(cond_vec.to(x_t.dtype))
        emb  = torch.cat([temb, cemb], dim=1).to(x_t.dtype)
        x0 = self.stem(x_in)
        d1 = self.down1(x0,emb); x = self.ds1(d1)
        d2 = self.down2(x,emb);  x = self.ds2(d2)
        x  = self.mid1(x,emb);   x = self.mid2(x,emb)
        x  = self.up2(x); x = torch.cat([x,d2], dim=1); x = self.upb2(x,emb)
        x  = self.up1(x); x = torch.cat([x,d1], dim=1); x = self.upb1(x,emb)
        return self.out(x)

class GlobalPredictor(nn.Module):
    def __init__(self, in_dim=7, hid=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, 64), nn.SiLU(),
        )
        self.head_ps = nn.Linear(64, 1)
        self.head_xy = nn.Linear(64, 2)
    def forward(self, x):
        h = self.backbone(x)
        ps_n = self.head_ps(h)
        xy   = torch.sigmoid(self.head_xy(h))
        return ps_n, xy

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in unwrap(model).state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        base = unwrap(model)
        for k, v in base.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def copy_to(self, model):
        unwrap(model).load_state_dict(self.shadow, strict=True)

# -----------------------
# UTIL: argmax XY (unit)
# -----------------------
def argmax_xy_unit_np(batch_np: np.ndarray):
    N,H,W = batch_np.shape
    flat_idx = batch_np.reshape(N,-1).argmax(axis=1)
    ys = (flat_idx // W).astype(np.float32)
    xs = (flat_idx %  W).astype(np.float32)
    return (xs / np.float32(W - 1)).astype(np.float32), (ys / np.float32(H - 1)).astype(np.float32)

# -----------------------
# SCAN SHARDS & STATS (streaming)
# -----------------------
def list_train_shards(data_dir: str):
    d = Path(data_dir) / "train"
    files = sorted(list(d.glob("train_shard_*.pkl")) + list(d.glob("train_shard_*.plk")))
    if not files:
        raise FileNotFoundError(f"Nessuno shard in {d}")
    return files

def load_test_df(data_dir: str):
    p = Path(data_dir) / "test.pkl"
    if not p.is_file():
        # anche .plk se serve
        alt = Path(data_dir) / "test.plk"
        if not alt.is_file():
            raise FileNotFoundError(f"Manca test.pkl in {data_dir}")
        p = alt
    return pd.read_pickle(p)

def infer_columns(df: pd.DataFrame):
    """Restituisce (cond7_cols, psum_col, image_col). Supporta sia schema per indici sia 'image' nominata."""
    cols = list(df.columns)
    if "image" in cols:
        image_col = "image"
        # cerca psum: nomi tipici
        for cand in ["photonSum", "psum", "PhotonSum", "psum_total"]:
            if cand in cols:
                psum_col = cand
                break
        else:
            # fallback: 9a colonna (indice 8)
            psum_col = cols[8]
        # cond7: 7 colonne numeriche; spesso 1..7 (indici 1..7)
        cond7_cols = cols[1:8]
    else:
        # fallback schema originale: 2..8 -> cond7, 9 -> psum, 10 -> matrice
        cond7_cols = cols[1:8]
        psum_col   = cols[8]
        image_col  = cols[9]
    return cond7_cols, psum_col, image_col

def streaming_stats(shards, seed=SEED):
    """
    Calcola mean/std di cond7 e di log1p(psum) in streaming (Welford).
    Non stima y_clip_max: lo fisseremo a 1024.
    """
    rng = np.random.RandomState(seed)
    n = 0
    mean_c = np.zeros((COND7_DIM,), dtype=np.float64)
    m2_c   = np.zeros((COND7_DIM,), dtype=np.float64)

    mean_ps = 0.0
    m2_ps   = 0.0

    for sh in shards:
        df = pd.read_pickle(sh)
        cond7_cols, psum_col, image_col = infer_columns(df)

        c  = df[cond7_cols].to_numpy(dtype=np.float32, copy=False)
        ps = df[psum_col].to_numpy(dtype=np.float32, copy=False)

        # Welford cond7
        for i in range(c.shape[0]):
            n += 1
            x = c[i].astype(np.float64, copy=False)
            delta = x - mean_c
            mean_c += delta / n
            m2_c   += delta * (x - mean_c)

        # Welford su log1p(psum)
        ps_log = np.log1p(ps, dtype=np.float64)
        for val in ps_log:
            delta = val - mean_ps
            mean_ps += delta / n
            m2_ps   += delta * (val - mean_ps)

        del df, c, ps, ps_log

    var_c = m2_c / max(1, n-1)
    std_c = np.sqrt(np.maximum(var_c, 1e-12))
    var_ps = m2_ps / max(1, n-1)
    std_ps = float(np.sqrt(max(var_ps, 1e-12)))

    return mean_c.astype(np.float32), std_c.astype(np.float32), float(mean_ps), float(std_ps)

# -----------------------
# ENCODE/DECODE target
# -----------------------
# placeholder; valori reali verranno settati dopo la stima
y_clip_max = None
y_log_scale = None

def encode_target(y_real_np: np.ndarray) -> np.ndarray:
    # y_real_np in uint16/float, shape [N,44,44]
    y_clip = np.clip(y_real_np, 0, y_clip_max).astype(np.float32, copy=False)
    y_log  = np.log1p(y_clip, dtype=np.float32)
    y_01   = y_log / np.float32(y_log_scale)
    y_mm1  = (np.float32(2.0) * y_01 - np.float32(1.0)).astype(np.float32, copy=False)
    return y_mm1

def decode_target_torch(y_mm1: torch.Tensor) -> torch.Tensor:
    y_mm1 = y_mm1.clamp(min=y_mm1.new_tensor(-MM1_CLAMP_TRAIN), max=y_mm1.new_tensor(MM1_CLAMP_TRAIN))
    y_01  = (y_mm1 + y_mm1.new_tensor(1.0)) * y_mm1.new_tensor(0.5)
    y_log = y_01 * y_mm1.new_tensor(y_log_scale, dtype=y_mm1.dtype)
    out   = torch.expm1(y_log)
    return torch.nan_to_num(out, nan=NAN_REPLACE_VAL, posinf=1e6, neginf=0.0)

# -----------------------
# DATASET SHARDATO (Iterable)
# -----------------------
class ShardedIterable(IterableDataset):
    """
    Itera su più shard (ordine random per epoca), caricando uno shard per volta.
    Applica:
      - cast immagine a uint16 (se non lo è)
      - normalizzazioni cond7/psum usando mean/std globali
      - encoding y -> [-1,1] (mm1)
    Ritorna tuple di tensori CPU (DataLoader farà pin & batch).
    """
    def __init__(self, shards, cond7_mean, cond7_std, psum_mean, psum_std, shuffle_buffer=10000, seed=SEED):
        super().__init__()
        self.shards = list(shards)
        self.cond7_mean = cond7_mean.astype(np.float32)
        self.cond7_std  = cond7_std.astype(np.float32)
        self.psum_mean  = float(psum_mean)
        self.psum_std   = float(psum_std)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = seed

    def _iter_worker(self, shards_subset):
        rng = np.random.RandomState(self.seed + (get_worker_info().id if get_worker_info() else 0))
        buffer = []

        for sh in shards_subset:
            df = pd.read_pickle(sh)
            cond7_cols, psum_col, image_col = infer_columns(df)

            # numpy view senza copie inutili
            cond7 = df[cond7_cols].to_numpy(dtype=np.float32, copy=False)
            psum  = df[psum_col].to_numpy(dtype=np.float32, copy=False)
            imgs  = df[image_col].tolist()  # liste di np.array

            # immagine -> uint16 garantito
            mats = np.stack([np.asarray(x, dtype=np.uint16) for x in imgs], axis=0)  # [N,44,44]

            # xy (unit) da argmax (vectorized)
            x_u, y_u = argmax_xy_unit_np(mats.astype(np.float32, copy=False))

            # normalizzazioni
            cond7_n = (cond7 - self.cond7_mean[None,:]) / (self.cond7_std[None,:] + 1e-6)
            ps_log  = np.log1p(psum, dtype=np.float32)
            ps_n    = (ps_log - np.float32(self.psum_mean)) / (np.float32(self.psum_std) + 1e-6)

            # encode target una volta sola (float32)
            y_mm1 = encode_target(mats)  # np.float32 [-1,1]
            # push nel buffer come tensori CPU (collate default ok)
            for i in range(len(mats)):
                item = (
                    torch.from_numpy(y_mm1[i]).unsqueeze(0),                # x0 (mm1) [1,44,44] float32
                    torch.from_numpy(cond7_n[i]).view(COND7_DIM),           # cond7_n float32
                    torch.tensor(ps_n[i], dtype=torch.float32).view(1),     # psum_n float32 [1]
                    torch.tensor([x_u[i], y_u[i]], dtype=torch.float32),    # xy_u float32 [2]
                    torch.tensor(psum[i], dtype=torch.float32),             # psum_real float32
                )
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer:
                    j = rng.randint(0, len(buffer))
                    yield buffer.pop(j)

            # svuota il buffer a fine shard
            rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()

            del df, cond7, psum, imgs, mats, x_u, y_u, cond7_n, ps_log, ps_n, y_mm1

    def __iter__(self):
        # Distribuisci shard tra i worker
        info = get_worker_info()
        if info is None:
            # singolo worker
            order = list(self.shards)
            random.Random(self.seed).shuffle(order)
            return self._iter_worker(order)
        else:
            n = info.num_workers
            wid = info.id
            order = list(self.shards)
            random.Random(self.seed + wid).shuffle(order)
            # partiziona round-robin
            subset = [order[i] for i in range(len(order)) if i % n == wid]
            return self._iter_worker(subset)

# -----------------------
# HELPERS FISICI + INTERI
# -----------------------
def project_positive_with_sum(x_real: torch.Tensor, sum_target: torch.Tensor, eps: float = 1e-6):
    x_pos = torch.relu(x_real)
    s = x_pos.flatten(1).sum(dim=1, keepdim=True)
    eps_t = torch.as_tensor(eps, device=x_real.device, dtype=x_real.dtype)
    scale = (sum_target.view(-1,1).to(x_real.dtype) / (s + eps_t)).clamp(min=x_real.new_tensor(0.0), max=x_real.new_tensor(1e6))
    out = x_pos * scale.view(-1,1,1,1)
    return torch.nan_to_num(out, nan=NAN_REPLACE_VAL, posinf=1e6, neginf=0.0)

def centroid_xy(x_real: torch.Tensor):
    B,_,H,W = x_real.shape
    ys = torch.linspace(0, 1, steps=H, device=x_real.device, dtype=x_real.dtype)
    xs = torch.linspace(0, 1, steps=W, device=x_real.device, dtype=x_real.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    mass = x_real.flatten(1).sum(dim=1) + x_real.new_tensor(1e-6)
    cx = (x_real * grid_x[None,None]).flatten(1).sum(dim=1) / mass
    cy = (x_real * grid_y[None,None]).flatten(1).sum(dim=1) / mass
    return cx, cy

def p2_weight(t_idx: torch.LongTensor):
    ab = alpha_bars.gather(0, t_idx).clamp(1e-8, 1-1e-8)
    snr = ab / (1.0 - ab)
    return (P2_K + snr) ** (-P2_GAMMA)

@torch.no_grad()
def integerize_with_sum(x_real: torch.Tensor, sum_int: torch.Tensor):
    B,_,H,W = x_real.shape
    flat = x_real.flatten(1)
    s = flat.sum(dim=1, keepdim=True) + flat.new_tensor(1e-8)
    p = (flat / s).clamp(min=0.0)
    target = sum_int.view(-1,1).to(p.dtype)
    expect = p * target
    y = expect.floor()
    rem = (target - y.sum(dim=1, keepdim=True)).squeeze(1)
    frac = (expect - y)
    y_int = y.to(torch.long)
    P = H*W
    for b in range(B):
        r = int(max(0, min(P, rem[b].item())))
        if r > 0:
            _, idx = torch.topk(frac[b], k=r)
            y_int[b, idx] += 1
    out = y_int.view(B,1,H,W)
    return out

# -----------------------
# SAMPLER (DDIM + CFG + proiezione fisica)
# -----------------------
@torch.no_grad()
def ddim_sample_cfg(unet_model, cond_vec, x_u, y_u, psum_pred_real, steps=EVAL_STEPS, cfg_scale=CFG_SCALE_EVAL, eta=0.0):
    base = unwrap(unet_model); base.eval()
    B = cond_vec.size(0); H = W = IMG_SIZE
    x = torch.randn(B,1,H,W, device=DEVICE).contiguous(memory_format=torch.channels_last)
    seq = torch.linspace(T_STEPS-1, 0, steps, device=DEVICE, dtype=torch.long)
    zeros_cond = torch.zeros_like(cond_vec)
    prior = gaussian_prior(x_u, y_u, H, W, PRIOR_SIGMA_PX, DEVICE, x.dtype)

    for j, t in enumerate(seq):
        t = t.long()
        t_prev = seq[j+1].long() if j+1 < len(seq) else None
        sqrt_ab_t, sqrt_1mab_t, ab_t = gather_ab(t.expand(B))
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(torch.cuda.is_available() and not args.no_amp)):
            eps_u = base(x, t.expand(B), zeros_cond, prior)
            eps_c = base(x, t.expand(B), cond_vec,   prior)
            eps   = eps_u + cfg_scale * (eps_c - eps_u)

        x0_mm1 = (x - sqrt_1mab_t.to(x.dtype) * eps) / (sqrt_ab_t.to(x.dtype) + x.new_tensor(1e-8))
        x0_mm1 = x0_mm1.clamp(min=x.new_tensor(-MM1_CLAMP_EVAL), max=x.new_tensor(MM1_CLAMP_EVAL))
        x0_real = decode_target_torch(x0_mm1)
        x0_proj = project_positive_with_sum(x0_real, psum_pred_real)
        x0_proj = x0_proj.clamp(min=x0_proj.new_tensor(0.0), max=x0_proj.new_tensor(y_clip_max))
        x0_mm1_proj = (torch.log1p(x0_proj) / x0_proj.new_tensor(y_log_scale)) * x0_proj.new_tensor(2.0) - x0_proj.new_tensor(1.0)

        if t_prev is None:
            x = x0_mm1_proj
            break

        _, _, ab_prev = gather_ab(t_prev.expand(B))
        sqrt_ab_prev   = torch.sqrt(ab_prev)
        sqrt_1mab_prev = torch.sqrt(1.0 - ab_prev)
        if eta == 0.0:
            x = sqrt_ab_prev.to(x.dtype) * x0_mm1_proj + sqrt_1mab_prev.to(x.dtype) * eps
        else:
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
            sigma = sigma.view(B,1,1,1).to(x.dtype)
            noise = torch.randn_like(x)
            x = sqrt_ab_prev.to(x.dtype) * x0_mm1_proj + torch.sqrt((sqrt_1mab_prev.to(x.dtype)**2 - sigma**2).clamp_min(0)) * eps + sigma * noise

        x = torch.nan_to_num(x, nan=NAN_REPLACE_VAL, posinf=1e3, neginf=-1e3)

    return x  # [-1,1]

# -----------------------
# SETUP MODELLI / OPT / EMA
# -----------------------
unet = CondUNet(cond_dim=COND_DIM, time_dim=TIME_DIM, cond_emb=COND_EMB, base_ch=BASE_CH).to(DEVICE)
pred = GlobalPredictor(in_dim=COND7_DIM, hid=128).to(DEVICE)
unet = unet.to(memory_format=torch.channels_last)

if not args.no_compile:
    try:
        unet = torch.compile(unet, mode="default", fullgraph=False)
        pred = torch.compile(pred, mode="default", fullgraph=False)
    except Exception:
        pass

ema_unet = EMA(unet, decay=EMA_DECAY)
ema_pred = EMA(pred, decay=EMA_DECAY)

opt = optim.AdamW(list(unet.parameters()) + list(pred.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
mse_pix = nn.MSELoss(reduction='none')
mse = nn.MSELoss()
SCALER = torch.amp.GradScaler('cuda', enabled=(torch.cuda.is_available() and not args.no_amp))

# Warm start opzionale
if args.use_best and Path("best_ema_checkpoint.pt").is_file():
    try:
        ckpt = torch.load("best_ema_checkpoint.pt", map_location=DEVICE)
        ema_unet.shadow = {k: v.detach().clone().to(DEVICE) for k, v in ckpt["unet_ema"].items()}
        ema_pred.shadow = {k: v.detach().clone().to(DEVICE) for k, v in ckpt["pred_ema"].items()}
        ema_unet.copy_to(unet); ema_pred.copy_to(pred)
        print("[WARM] Caricati EMA (UNet + Predictor) da best_ema_checkpoint.pt")
    except Exception as e:
        print(f"[WARM][ERROR] {e}")

# -----------------------
# PREPARA DATI: STATS + DATASET + DATALOADER
# -----------------------
all_shards = list_train_shards(DATA_DIR)
if args.train_shards is not None:
    if args.train_shards <= 0:
        raise ValueError("--train-shards deve essere > 0")
    # Limita il set *di base*; l'ordine verrà randomizzato per epoca comunque
    train_shards = all_shards[:args.train_shards]
else:
    train_shards = all_shards

print(f"[DATA] Shard di train: {len(train_shards)} / tot={len(all_shards)}")



# Statistiche streaming sui soli shard selezionati
cond7_mean, cond7_std, psum_mean, psum_std = streaming_stats(train_shards, seed=SEED)
# Clip fisso
y_clip_max = float(Y_CLIP_MAX_DEFAULT)
y_log_scale = float(np.log1p(np.float32(y_clip_max)))
print(f"[STATS] cond7_mean shape={cond7_mean.shape} | psum_mean={psum_mean:.4f} | y_clip_max={y_clip_max:.1f} (fixed)")

# Dataset/Dataloader
train_ds = ShardedIterable(train_shards, cond7_mean, cond7_std, psum_mean, psum_std,
                           shuffle_buffer=args.shuffle_buffer, seed=SEED)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=args.num_workers,
    pin_memory=True,
    prefetch_factor=args.prefetch_factor,
    persistent_workers=(args.num_workers > 0),
    drop_last=True
)

# Test set (carichiamo una volta; useremo solo EVAL_NUM esempi)
test_df = load_test_df(DATA_DIR)
cond7_cols_te, psum_col_te, image_col_te = infer_columns(test_df)
cond7_te_np = test_df[cond7_cols_te].to_numpy(dtype=np.float32, copy=False)
psum_te_np  = test_df[psum_col_te].to_numpy(dtype=np.float32, copy=False)
imgs_te     = [np.asarray(x, dtype=np.uint16) for x in test_df[image_col_te].tolist()]
mats_te_np  = np.stack(imgs_te, axis=0)  # uint16
x_te_u, y_te_u = argmax_xy_unit_np(mats_te_np.astype(np.float32, copy=False))

cond7_te_n = (cond7_te_np - cond7_mean[None,:]) / (cond7_std[None,:] + 1e-6)
psum_te_log = np.log1p(psum_te_np, dtype=np.float32)
psum_te_n   = (psum_te_log - np.float32(psum_mean)) / (np.float32(psum_std) + 1e-6)
Y_te_mm1_np = encode_target(mats_te_np)  # float32 [-1,1]

# Tensori test su DEVICE solo quando servono (eval prende subset)
cond7_te_t_all = torch.from_numpy(cond7_te_n).to(DEVICE)
psum_te_n_t_all= torch.from_numpy(psum_te_n).to(DEVICE).view(-1,1)
xy_te_u_t_all  = torch.from_numpy(np.stack([x_te_u,y_te_u],1).astype(np.float32)).to(DEVICE)
psum_te_real_t_all = torch.from_numpy(psum_te_np.astype(np.float32)).to(DEVICE)
Yte_mm1_all    = torch.from_numpy(Y_te_mm1_np).unsqueeze(1).to(DEVICE).contiguous(memory_format=torch.channels_last)
Yte_raw_int_all= torch.from_numpy(mats_te_np.astype(np.int64)).unsqueeze(1).to(DEVICE)  # per int_eval

# -----------------------
# TRAIN / EVAL
# -----------------------
def train_epoch(epoch_idx=None):

    unet.train(); pred.train()
    total = 0.0; nb = 0
    total_eps = 0.0; total_phys=0.0; total_cent=0.0
    total_ps=0.0; total_xy=0.0
    total_psum_mse = 0.0
    total_center_arg = 0.0
    total_gmax = 0.0
    total_entropy = 0.0

    warned = False

    for (x0_cpu, cond7_cpu, psum_n_cpu, xy_u_cpu, psum_real_cpu) in train_loader:
        # trasferimenti (overlap con pin_memory+non_blocking)
        x0 = x0_cpu.to(DEVICE, non_blocking=True)               # [B,1,44,44] float32 in [-1,1]
        cond7 = cond7_cpu.to(DEVICE, non_blocking=True)         # [B,7]
        psum_n = psum_n_cpu.to(DEVICE, non_blocking=True)       # [B,1]
        xy_u = xy_u_cpu.to(DEVICE, non_blocking=True)           # [B,2]
        ps_real = psum_real_cpu.to(DEVICE, non_blocking=True)   # [B]

        Bcur = x0.size(0)

        ps_pred_n, xy_pred_u = pred(cond7)
        use_gt_mask = (torch.rand(Bcur, device=DEVICE) < args.teacher_p).float().view(-1,1)
        ps_for_B_n  = use_gt_mask * psum_n + (1.0 - use_gt_mask) * ps_pred_n.detach()
        xy_for_B_u  = use_gt_mask * xy_u     + (1.0 - use_gt_mask) * xy_pred_u.detach()

        cond_vec = torch.cat([cond7, ps_for_B_n, xy_for_B_u], dim=1).to(torch.float32)

        t = torch.randint(0, T_STEPS, (Bcur,), device=DEVICE, dtype=torch.long)
        sqrt_ab_t, sqrt_1mab_t, _ = gather_ab(t)
        eps = torch.randn_like(x0)
        x_t = sqrt_ab_t.to(x0.dtype) * x0 + sqrt_1mab_t.to(x0.dtype) * eps

        prior = gaussian_prior(xy_for_B_u[:,0], xy_for_B_u[:,1], IMG_SIZE, IMG_SIZE, PRIOR_SIGMA_PX, DEVICE, x_t.dtype)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(torch.cuda.is_available() and not args.no_amp)):
            eps_pred = unet(x_t, t, cond_vec, prior)

            # epsilon loss con pesi di sparsità (in spazio reale)
            x0_real = decode_target_torch(x0)
            w_pix = 1.0 + SPARSE_LAMBDA * (x0_real > x0_real.new_tensor(SPARSE_THRESH)).float()
            per_pix = mse_pix(eps_pred, eps) * w_pix
            per_img = per_pix.mean(dim=(1,2,3))
            w_t = p2_weight(t)
            loss_eps = (per_img * w_t).mean()

            # Physics: somma/centroidi su x0_hat con proiezione a ps_real (vincolo fotoni)
            x0_hat_mm1  = (x_t - sqrt_1mab_t.to(x_t.dtype) * eps_pred) / (sqrt_ab_t.to(x_t.dtype) + x_t.new_tensor(1e-8))
            x0_hat_mm1  = x0_hat_mm1.clamp(min=x0_hat_mm1.new_tensor(-MM1_CLAMP_TRAIN), max=x0_hat_mm1.new_tensor(MM1_CLAMP_TRAIN))
            x0_hat_real = decode_target_torch(x0_hat_mm1)
            x0_proj     = project_positive_with_sum(x0_hat_real, ps_real)
            loss_phys   = mse(x0_proj, x0_real)

            cx_pred, cy_pred = centroid_xy(x0_proj)
            cx_true, cy_true = centroid_xy(x0_real)
            loss_cent = mse(cx_pred, cx_true) + mse(cy_pred, cy_true)
            # ---------- NUOVE LOSS "real-aligned" ----------
            # x0_real:     reale (decode di target)   [B,1,H,W] >= 0
            # x0_hat_real: predizione prima della proiezione
            # x0_proj:     predizione PROIETTATA a ps_real (positività + somma vincolata)

            # 1) Photon sum (MSE) — confronta le somme PRIMA della proiezione (ha senso per dare learning-signal)
            sum_real = gamma_sum_torch(x0_real)
            sum_pred_unproj = gamma_sum_torch(x0_hat_real)
            loss_psum_mse = mse(sum_pred_unproj, sum_real)

            # 2) Centro "tipo argmax" (differenziabile via soft-argmax) — meglio confrontare DOPO proiezione
            cx_r, cy_r = soft_argmax_xy_unit(x0_real, tau=CENTER_TAU)
            cx_p, cy_p = soft_argmax_xy_unit(x0_proj,  tau=CENTER_TAU)
            loss_center_arg = mse(cx_p, cx_r) + mse(cy_p, cy_r)

            # 3) Valore massimo (gamma_max) — MSE; gradiente fluisce sul pixel massimo
            gmax_r = gamma_max_torch(x0_real)
            gmax_p = gamma_max_torch(x0_proj)
            loss_gmax = mse(gmax_p, gmax_r)

            # 4) Entropia normalizzata — MSE; misura “concentrazione” dello shower
            ent_r = entropy_normalized_torch(x0_real)
            ent_p = entropy_normalized_torch(x0_proj)
            loss_entropy = mse(ent_p, ent_r)
            # -----------------------------------------------

            # Regressione stadio A
            loss_ps = mse(ps_pred_n, psum_n)
            loss_xy = mse(xy_pred_u, xy_u)

            loss = (
                loss_eps
                + LAMBDA_SUM*loss_phys + LAMBDA_SUM*loss_cent
                + LAMBDA_REG_PSUM*loss_ps + LAMBDA_REG_XY*loss_xy
                # nuovi termini
                + LAMBDA_PSUM_MSE      * loss_psum_mse
                + LAMBDA_CENTER_ARGMAX * loss_center_arg
                + LAMBDA_GMAX          * loss_gmax
                + LAMBDA_ENTROPY       * loss_entropy
            )


        if not torch.isfinite(loss):
            if not warned:
                print("[WARN] loss non finita: skip batch")
                warned = True
            continue

        if SCALER.is_enabled():
            SCALER.scale(loss).backward()
            if GRAD_CLIP:
                SCALER.unscale_(opt)
                nn.utils.clip_grad_norm_(list(unet.parameters()) + list(pred.parameters()), GRAD_CLIP)
            SCALER.step(opt); SCALER.update()
        else:
            loss.backward()
            if GRAD_CLIP: nn.utils.clip_grad_norm_(list(unet.parameters()) + list(pred.parameters()), GRAD_CLIP)
            opt.step()

        ema_unet.update(unet); ema_pred.update(pred)
        total += float(loss.detach())*Bcur; nb += Bcur
        total_eps += float(loss_eps.detach())*Bcur
        total_cent += float(loss_cent.detach())*Bcur
        total_phys += float(loss_phys.detach())*Bcur
        total_ps += float(loss_ps.detach())*Bcur
        total_xy += float(loss_xy.detach())*Bcur
        total_psum_mse += float(loss_psum_mse.detach())*Bcur
        total_center_arg += float(loss_center_arg.detach())*Bcur
        total_gmax += float(loss_gmax.detach())*Bcur
        total_entropy += float(loss_entropy.detach())*Bcur
    nb = max(1, nb)
    total_xy /= nb
    total_cent/=nb
    total_eps/=nb
    total_phys/=nb
    total_ps/=nb
    total /= nb
    total_psum_mse /= nb
    total_center_arg /= nb
    total_gmax /= nb
    total_entropy /= nb
    #Scrittura delle compponenti della loss
    file_exists= os.path.isfile("loss_component_all.csv")
    with open("loss_component_all.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "epoch","loss_total","loss_eps","loss_phys","loss_cent",
                "loss_ps","loss_xy",
                "loss_psum_mse","loss_center_arg","loss_gmax","loss_entropy"
            ])
        writer.writerow([
            (int(epoch_idx) if epoch_idx is not None else ""),
            total, total_eps, total_phys, total_cent,
            total_ps, total_xy,
            total_psum_mse, total_center_arg, total_gmax, total_entropy
        ])

    return total

@torch.no_grad()
def eval_end_to_end(ema_u_state, ema_p_state, split="test", num=EVAL_NUM, steps=EVAL_STEPS, cfg_scale=CFG_SCALE_EVAL, int_eval=False):
    # ricreo i modelli EMA
    u = CondUNet(cond_dim=COND_DIM, time_dim=TIME_DIM, cond_emb=COND_EMB, base_ch=BASE_CH).to(DEVICE)
    p = GlobalPredictor(in_dim=COND7_DIM, hid=128).to(DEVICE)
    u = u.to(memory_format=torch.channels_last)
    u.load_state_dict(ema_u_state, strict=True); u.eval()
    p.load_state_dict(ema_p_state, strict=True); p.eval()

    if split == "test":
        idx = torch.arange(min(num, Yte_mm1_all.size(0)), device=DEVICE)
        cond7 = cond7_te_t_all[idx]
        real_mm1 = Yte_mm1_all[idx]
        psum_real = psum_te_real_t_all[idx]
        real_int = Yte_raw_int_all[idx]
    else:
        # per semplicità: campiona dal test anche per 'train' display (potresti aggiungere un mini-cache train se vuoi)
        idx = torch.arange(min(num, Yte_mm1_all.size(0)), device=DEVICE)
        cond7 = cond7_te_t_all[idx]
        real_mm1 = Yte_mm1_all[idx]
        psum_real = psum_te_real_t_all[idx]
        real_int = Yte_raw_int_all[idx]

    # Step A: predizione globale
    ps_pred_n, xy_pred_u = p(cond7)
    psum_pred_real = torch.expm1(ps_pred_n.squeeze(1) * cond7.new_tensor(psum_std) + cond7.new_tensor(psum_mean))
    psum_pred_real = torch.nan_to_num(psum_pred_real, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=0.0)

    # Condizione per UNet
    cond_vec = torch.cat([cond7, ps_pred_n, xy_pred_u], dim=1).to(torch.float32)

    # Step B: sampling
    fake_mm1 = ddim_sample_cfg(u, cond_vec, xy_pred_u[:,0], xy_pred_u[:,1], psum_pred_real,
                               steps=steps, cfg_scale=cfg_scale, eta=0.0)
    fake_real = decode_target_torch(fake_mm1).clamp(min=0.0, max=y_clip_max)

    if int_eval:
        psum_pred_int = torch.round(psum_pred_real).clamp(min=0).to(torch.long)
        fake_int = integerize_with_sum(fake_real, psum_pred_int)
        # metriche intere
        diff2 = (fake_int.float() - real_int.float()) ** 2
        mse_per_pixel = diff2.mean().item()
        sse = diff2.sum().item()
        rmse = float(mse_per_pixel ** 0.5)
        frac_zero_real = (real_int <= 0).float().mean().item()
        frac_zero_fake = (fake_int <= 0).float().mean().item()
        # errore sulla somma fotoni (dopo integerize, deve essere ~0)
        sum_real = real_int.flatten(1).sum(dim=1).float()
        sum_fake = fake_int.flatten(1).sum(dim=1).float()
        mean_abs_sum_err = (sum_fake - sum_real).abs().mean().item()
        return mse_per_pixel, rmse, sse, frac_zero_real, frac_zero_fake, mean_abs_sum_err
    else:
        real_real = decode_target_torch(real_mm1).clamp(min=0.0, max=y_clip_max)
        diff2 = (fake_real - real_real) ** 2
        mse_per_pixel = diff2.mean().item()
        sse = diff2.sum().item()
        rmse = float(mse_per_pixel ** 0.5)
        frac_zero_real = (real_real <= SPARSE_THRESH).float().mean().item()
        frac_zero_fake = (fake_real <= SPARSE_THRESH).float().mean().item()
        # errore sulla somma fotoni (in continuo, solo diagnostica)
        sum_real = real_real.flatten(1).sum(dim=1)
        sum_fake = fake_real.flatten(1).sum(dim=1)
        mean_abs_sum_err = (sum_fake - sum_real).abs().mean().item()
        return mse_per_pixel, rmse, sse, frac_zero_real, frac_zero_fake, mean_abs_sum_err

# -----------------------
# TRAIN LOOP
# -----------------------
best_mse = float("inf")
P = IMG_SIZE * IMG_SIZE

for ep in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss = train_epoch(ep)

    ema_u = {k: v.clone() for k, v in ema_unet.shadow.items()}
    ema_p = {k: v.clone() for k, v in ema_pred.shadow.items()}

    test_pp, test_rmse, test_sse, z_real_t, z_fake_t, sum_err_t = eval_end_to_end(
        ema_u, ema_p, split="test", int_eval=args.int_eval)
    train_pp, train_rmse, train_sse, z_real_tr, z_fake_tr, sum_err_tr = eval_end_to_end(
        ema_u, ema_p, split="train", int_eval=args.int_eval)

    dt = time.time() - t0
    print(f"[Epoch {ep:03d}] loss={tr_loss:.5f} | "
          f"TEST: MSEpp={test_pp:.4f} RMSE={test_rmse:.4f} SSE={test_sse:.1f} ≈ {test_pp:.2f}*{P} "
          f"zeros(real)={z_real_t:.3f} zeros(fake)={z_fake_t:.3f} Δpsum|mean|={sum_err_t:.3f} | "
          f"TRAIN: MSEpp={train_pp:.4f} RMSE={train_rmse:.4f} SSE={train_sse:.1f} "
          f"zeros(real)={z_real_tr:.3f} zeros(fake)={z_fake_tr:.3f} Δpsum|mean|={sum_err_tr:.3f} | "
          f"{dt:.1f}s")

    ckpt_dir = Path("DDPM_stage_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    epoch_ckpt = {
        "epoch": ep,
        "unet_ema": ema_u,
        "pred_ema": ema_p,
        "cond7_mean": torch.from_numpy(cond7_mean).float(),
        "cond7_std":  torch.from_numpy(cond7_std).float(),
        "psum_mean": float(psum_mean),
        "psum_std":  float(psum_std),
        "y_clip_max": float(y_clip_max),
        "y_log_scale": float(y_log_scale),
        "cfg": {
            "IMG_SIZE": IMG_SIZE, "COND7_DIM": COND7_DIM,
            "TIME_DIM": TIME_DIM, "COND_EMB": COND_EMB, "BASE_CH": BASE_CH,
            "T_STEPS": T_STEPS, "PRIOR_SIGMA_PX": PRIOR_SIGMA_PX
        }
    }
    torch.save(epoch_ckpt, ckpt_dir / f"epoch_{ep:03d}.pt")

    if test_pp < best_mse and np.isfinite(test_pp):
        best_mse = test_pp
        torch.save(epoch_ckpt, "best_ema_checkpoint.pt")
        print(f"  -> salvato best_ema_checkpoint.pt (best MSE={best_mse:.4f})")

    # CSV metrics (append)
    metrics_path = Path("DDPM_stage_metrics.csv")
    log_row = {
        "epoch": ep,
        "train_total_loss": float(tr_loss),
        "test_mse_pp": float(test_pp),
        "test_rmse": float(test_rmse),
        "test_sse": float(test_sse),
        "train_mse_pp": float(train_pp),
        "train_rmse": float(train_rmse),
        "train_sse": float(train_sse),
        "zeros_real_test": float(z_real_t),
        "zeros_fake_test": float(z_fake_t),
        "zeros_real_train": float(z_real_tr),
        "zeros_fake_train": float(z_fake_tr),
        "mean_abs_sum_err_test": float(sum_err_t),
        "mean_abs_sum_err_train": float(sum_err_tr),
        "lr": float(opt.param_groups[0]["lr"]),
        "eval_steps": int(EVAL_STEPS),
        "cfg_scale_eval": float(CFG_SCALE_EVAL),
        "teacher_p": float(args.teacher_p),
        "epoch_time_sec": float(dt),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "shuffle_buffer": int(args.shuffle_buffer),
        "train_shards_used": int(len(train_shards)),
        "batch_size": int(BATCH_SIZE),
    }
    pd.DataFrame([log_row]).to_csv(
        metrics_path,
        mode="a",
        header=not metrics_path.exists(),
        index=False
    )

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print("Training finito.")

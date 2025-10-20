# ============================================================
# DDPM_shape_blr.py — Single-stage diffusion on SHAPE + BLR SCALE
#  • Diffusion su z (logit della forma p) con v-prediction
#  • Scala s campionata da Bayesian Linear Regression (Student-t)
#  • Loss fisiche su Y = s * softmax(z0) (data-consistent)
#  • Eval: shape-only (s reale) + full (s campionata)
#  • Veloce: 44x44, pochi step DDIM, fp16/channels_last/compile
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
# CONFIG / HYPERPARAMS
# -----------------------
DATA_DIR     = "/data/dataalice/dfuligno/tesi/myTesi/Dati_processed"
SEED         = 42
IMG_SIZE     = 44
BATCH_SIZE   = 64
EPOCHS       = 150
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EMA_DECAY    = 0.999
T_STEPS      = 1000

# Eval
EVAL_STEPS   = 8          # pochi step per velocità (8-12)
EVAL_NUM     = 32

# Modello
COND7_DIM    = 7
BASE_CH      = 48         # 32–64 bastano
TIME_DIM     = 64
COND_EMB     = 64
GRAD_CLIP    = 1.0

# CFG (classifier-free guidance)
CFG_P_UNCOND   = 0.05
CFG_SCALE_EVAL = 0.5

# P2 weighting
P2_K     = 1.0
P2_GAMMA = 0.5

# Physics loss weights (su Y = s * softmax(z0))
W_CENTER_PX    = 1.0
W_PEAK_LOG     = 1.0
W_SHAPE_RAD    = 0.3
CENTER_TAU     = 0.02
RADII_PX       = (1, 2, 3, 5)  # su immagine 22x22 (downsample x2)
RADIAL_SIGMOID_DELTA = 0.2

# Stabilità/encode
Z_CLAMP        = 8.0     # clamp di sicurezza su z
EPS_P          = 1e-6    # eps nella log/softmax della forma

# Stats cond7
COND_STD_FLOOR = 1e-2
COND_CLIP      = 9.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except: pass
AMP_DTYPE = torch.float16

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser(description="Single-stage DDPM (shape logits + BLR scale)")
parser.add_argument("--data-dir", type=str, default=DATA_DIR)
parser.add_argument("--train-shards", type=int, default=None)
parser.add_argument("--shuffle-buffer", type=int, default=10000)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--prefetch-factor", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--lr", type=float, default=LR)
parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
parser.add_argument("--no_compile", action="store_true")
parser.add_argument("--no_amp", action="store_true")
parser.add_argument("--int_eval", action="store_true", help="metriche intere con integerize(sum)")
parser.add_argument("--cfg_scale_eval", type=float, default=CFG_SCALE_EVAL)
parser.add_argument("--p_uncond", type=float, default=CFG_P_UNCOND)
parser.add_argument("--eval_steps", type=int, default=EVAL_STEPS)
parser.add_argument("--eval_debug", action="store_true")
parser.add_argument("--blr_clip_quantiles", type=float, nargs=2, default=[0.005, 0.995],
                    help="Clip dei campioni y=log1p(s) ai quantili empirici [low, high]")
args = parser.parse_args()

BATCH_SIZE      = args.batch_size
EPOCHS          = args.epochs
LR              = args.lr
WEIGHT_DECAY    = args.weight_decay
DATA_DIR        = args.data_dir
CFG_SCALE_EVAL  = args.cfg_scale_eval
CFG_P_UNCOND    = args.p_uncond
EVAL_STEPS      = args.eval_steps

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
# MODELLI (v-prediction)
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

class CondUNetV(nn.Module):
    """UNet che predice v_t su z-space (logit della forma). Input canali: [x_t, gx, gy]."""
    def __init__(self, cond_dim=7, time_dim=64, cond_emb=64, base_ch=48):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim,cond_emb), nn.SiLU(), nn.Linear(cond_emb,cond_emb))
        emb_dim = time_dim + cond_emb
        ch = base_ch
        self.stem = nn.Conv2d(3, ch, 3, padding=1)  # [x_t, gx, gy]
        self.down1 = ResBlock(ch, ch, emb_dim); self.ds1 = Downsample(ch)
        self.down2 = ResBlock(ch, ch*2, emb_dim); self.ds2 = Downsample(ch*2)
        self.mid1  = ResBlock(ch*2, ch*4, emb_dim); self.mid2 = ResBlock(ch*4, ch*4, emb_dim)
        self.up2   = Upsample(ch*4); self.upb2 = ResBlock(ch*4 + ch*2, ch*2, emb_dim)
        self.up1   = Upsample(ch*2); self.upb1 = ResBlock(ch*2 + ch, ch, emb_dim)
        self.out   = nn.Sequential(nn.GroupNorm(min(8,ch), ch), nn.SiLU(), nn.Conv2d(ch,1,3,padding=1))
    def forward(self, x_t, t, cond_vec):
        B,_,H,W = x_t.shape
        gx, gy = xy_grid(H, W, x_t.device, x_t.dtype)
        x_in = torch.cat([x_t, gx.expand(B,1,H,W), gy.expand(B,1,H,W)], dim=1)
        temb = self.time_emb(t)
        cemb = self.cond_mlp(cond_vec.to(x_t.dtype))
        emb  = torch.cat([temb, cemb], dim=1).to(x_t.dtype)
        x0 = self.stem(x_in)
        d1 = self.down1(x0,emb); x = self.ds1(d1)
        d2 = self.down2(x,emb);  x = self.ds2(d2)
        x  = self.mid1(x,emb);   x = self.mid2(x,emb)
        x  = self.up2(x); x = torch.cat([x,d2], dim=1); x = self.upb2(x,emb)
        x  = self.up1(x); x = torch.cat([x,d1], dim=1); x = self.upb1(x,emb)
        return self.out(x)   # v_t

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
# UTILS / DATA
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
        alt = Path(data_dir) / "test.plk"
        if not alt.is_file():
            raise FileNotFoundError(f"Manca test.pkl in {data_dir}")
        p = alt
    return pd.read_pickle(p)

def infer_columns(df: pd.DataFrame):
    cols = list(df.columns)
    if "image" in cols:
        image_col = "image"
        for cand in ["photonSum", "psum", "PhotonSum", "psum_total"]:
            if cand in cols: psum_col = cand; break
        else:
            psum_col = cols[8]
        cond7_cols = cols[1:8]
    else:
        cond7_cols = cols[1:8]; psum_col = cols[8]; image_col = cols[9]
    return cond7_cols, psum_col, image_col

# --- cond7 streaming stats + BLR sufficient statistics ---
def cond7_and_blr_stats(shards, seed=SEED):
    rng = np.random.RandomState(seed)
    # cond7 stats
    n_c = 0
    mean_c = np.zeros((COND7_DIM,), dtype=np.float64)
    m2_c   = np.zeros((COND7_DIM,), dtype=np.float64)
    # BLR stats (x=[1, cond7_n], y=log1p(psum))
    p = 1 + COND7_DIM
    XtX = np.zeros((p,p), dtype=np.float64)
    Xty = np.zeros((p,),   dtype=np.float64)
    yTy = 0.0
    N   = 0
    y_min = +1e9
    y_max = -1e9

    # Prima passata: cond7 mean/std
    for sh in shards:
        df = pd.read_pickle(sh)
        cond7_cols, psum_col, image_col = infer_columns(df)
        c  = df[cond7_cols].to_numpy(dtype=np.float32, copy=False)
        for i in range(c.shape[0]):
            n_c += 1
            x = c[i].astype(np.float64, copy=False)
            delta = x - mean_c
            mean_c += delta / n_c
            m2_c   += delta * (x - mean_c)

    var_c = m2_c / max(1, n_c-1)
    std_c = np.sqrt(np.maximum(var_c, 1e-12)).astype(np.float32)
    mean_c = mean_c.astype(np.float32)

    # Seconda passata: BLR sufficient stats (con cond7 normalizzate)
    for sh in shards:
        df = pd.read_pickle(sh)
        cond7_cols, psum_col, image_col = infer_columns(df)
        c   = df[cond7_cols].to_numpy(dtype=np.float32, copy=False)
        ps  = df[psum_col].to_numpy(dtype=np.float32, copy=False)
        c_n = (c - mean_c[None,:]) / (np.maximum(std_c[None,:], COND_STD_FLOOR))
        c_n = np.clip(c_n, -COND_CLIP, COND_CLIP)
        y   = np.log1p(ps, dtype=np.float64)

        # accumula sufficient stats
        ones = np.ones((c_n.shape[0],1), dtype=np.float32)
        X = np.concatenate([ones, c_n], axis=1).astype(np.float64, copy=False)  # [N, 1+7]
        XtX += X.T @ X
        Xty += X.T @ y
        yTy += float((y*y).sum())
        N   += X.shape[0]
        y_min = min(y_min, float(y.min()))
        y_max = max(y_max, float(y.max()))

    return mean_c, std_c, XtX, Xty, yTy, N, float(y_min), float(y_max)

# --- BLR posterior predictive (OLS+t) ---
class BLRScaleSampler:
    """
    Posterior predittivo t-Student su y=log1p(s) usando sufficient stats (XtX, Xty, yTy, N).
    Prior implicito di Jeffreys (equiv. OLS con varianza ignota).
    """
    def __init__(self, mean_c, std_c, XtX, Xty, yTy, N, y_min, y_max):
        self.mean_c = torch.tensor(mean_c, dtype=torch.float32)
        self.std_c  = torch.tensor(np.maximum(std_c, COND_STD_FLOOR), dtype=torch.float32)
        self.XtX = torch.tensor(XtX, dtype=torch.float64)
        self.Xty = torch.tensor(Xty, dtype=torch.float64)
        self.yTy = float(yTy)
        self.N   = int(N)
        self.p   = XtX.shape[0]
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        # beta_hat = (XtX)^-1 Xty  (pinv per sicurezza)
        self.XtX_inv = torch.linalg.pinv(self.XtX)  # [p,p] (float64)
        self.beta = (self.XtX_inv @ self.Xty)       # [p]   (float64)

        # SSE = y^T y - 2 beta^T X^T y + beta^T X^T X beta
        beta_col = self.beta.view(-1,1)
        SSE = self.yTy - 2.0*float((self.beta @ self.Xty).item()) + float((beta_col.T @ (self.XtX @ beta_col)).item())
        dof = max(1, self.N - self.p)
        self.sigma2 = max(1e-8, SSE / dof)
        self.dof = dof

    @torch.no_grad()
    def sample_s(self, cond7: torch.Tensor, clip_quantiles=(0.005, 0.995)) -> torch.Tensor:
        """
        cond7: [B,7] (float32) sul device corrente (cpu/cuda)
        ritorna s_hat (float32) >= 0 campionata dalla predittiva t-Student.
        """
        device = cond7.device

        # Normalizza cond7 sul device corrente
        c_n = (cond7 - self.mean_c.to(device)) / self.std_c.to(device)
        c_n = torch.clamp(c_n, -COND_CLIP, COND_CLIP)
        ones = torch.ones(c_n.size(0), 1, device=device, dtype=torch.float32)
        X = torch.cat([ones, c_n], dim=1).to(torch.float64)  # [B,p] in float64 per stabilità

        # Porta i parametri BLR sul device corrente (float64)
        beta     = self.beta.to(device)          # [p] float64
        XtX_inv  = self.XtX_inv.to(device)       # [p,p] float64
        sigma2   = torch.as_tensor(self.sigma2, dtype=torch.float64, device=device)

        # Media predittiva: m = X @ beta   -> [B] float64
        m = (X @ beta)

        # Varianza predittiva: v = sigma2 * (1 + diag(X @ XtX_inv @ X^T))
        # Implementazione efficiente: h_i = (X_i @ XtX_inv * X_i).sum()
        X_V   = X @ XtX_inv            # [B,p]
        h     = (X_V * X).sum(dim=1)   # [B]
        v     = sigma2 * (1.0 + h)
        v     = torch.clamp(v, min=torch.finfo(v.dtype).eps)

        # Parametri Student-t in float32 per torch.distributions
        loc   = m.to(torch.float32)
        scale = torch.sqrt(v).to(torch.float32)
        dist  = torch.distributions.StudentT(df=float(self.dof), loc=loc, scale=scale)

        # Campionamento su device corrente
        y = dist.sample()  # [B], y = log1p(s), float32 su 'device'

        # Clip a quantili empirici osservati (costanti portate sul device)
        q_low, q_high = clip_quantiles
        y_lo = torch.tensor(self.y_min + (self.y_max - self.y_min) * q_low,
                            device=device, dtype=y.dtype)
        y_hi = torch.tensor(self.y_min + (self.y_max - self.y_min) * q_high,
                            device=device, dtype=y.dtype)
        y = torch.clamp(y, min=y_lo, max=y_hi)

        s = torch.expm1(y)
        return torch.clamp(s, min=0.0)

# ------------- Encode/Decode SHAPE (logit space) -------------
def to_shape_logits(Y_hw: np.ndarray, eps=EPS_P, z_clamp=Z_CLAMP):
    """
    Y_hw: np.uint16/float32 [H,W], H=W=44, Y >=0, somma >0
    ritorna z [1,H,W] float32 (zero-mean log-prob)
    """
    Y = Y_hw.astype(np.float32, copy=False)
    s = float(Y.sum())
    if s <= 0.0: return None
    p = Y / s
    logp = np.log(p + float(eps), dtype=np.float32)
    logp -= float(logp.mean())
    logp = np.clip(logp, -float(z_clamp), float(z_clamp)).astype(np.float32)
    return logp[None, :, :]

@torch.no_grad()
def softmax_shape(z0: torch.Tensor, eps=EPS_P):
    """
    z0: [B,1,H,W] (float)
    ritorna p: [B,1,H,W], somma = 1
    """
    B,_,H,W = z0.shape
    z = z0 - z0.flatten(1).mean(dim=1, keepdim=True).view(B,1,1,1)  # recenter
    z = torch.clamp(z, min=-Z_CLAMP, max=Z_CLAMP)
    p = torch.softmax(z.flatten(1), dim=1).view(B,1,H,W)
    return p

def down2(x): return torch.nn.functional.avg_pool2d(x, 2, 2)

def soft_argmax_xy_unit_stable(x_real: torch.Tensor, tau: float = CENTER_TAU):
    # usa log1p per stabilità su skew elevato
    B,_,H,W = x_real.shape
    flat_log = torch.log1p(x_real.float()).flatten(1)  # [B,H*W]
    p = torch.softmax(flat_log / float(tau), dim=1)
    ys = torch.linspace(0, 1, steps=H, device=x_real.device, dtype=torch.float32)
    xs = torch.linspace(0, 1, steps=W, device=x_real.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    cx = (p * gx.reshape(1,-1)).sum(dim=1)
    cy = (p * gy.reshape(1,-1)).sum(dim=1)
    return cx, cy

def radial_cdf_loss(x_pred, x_real, cx_px, cy_px, radii=RADII_PX, delta=RADIAL_SIGMOID_DELTA):
    B,_,H,W = x_pred.shape
    xs = torch.arange(W, device=x_pred.device, dtype=x_pred.dtype).view(1,1,1,W).expand(B,1,H,W)
    ys = torch.arange(H, device=x_pred.device, dtype=x_pred.dtype).view(1,1,H,1).expand(B,1,H,W)
    ux = xs - cx_px.view(-1,1,1,1)
    vy = ys - cy_px.view(-1,1,1,1)
    r = torch.sqrt(ux**2 + vy**2)
    p_pred = x_pred / (x_pred.flatten(1).sum(dim=1, keepdim=True).view(B,1,1,1) + x_pred.new_tensor(1e-8))
    p_real = x_real / (x_real.flatten(1).sum(dim=1, keepdim=True).view(B,1,1,1) + x_real.new_tensor(1e-8))
    Cpred, Creal = [], []
    for R in radii:
        m = torch.sigmoid((float(R) - r) / float(delta))
        Cpred.append((m * p_pred).flatten(1).sum(dim=1))
        Creal.append((m * p_real).flatten(1).sum(dim=1))
    Cpred = torch.stack(Cpred, dim=1)
    Creal = torch.stack(Creal, dim=1)
    return torch.nn.functional.mse_loss(Cpred, Creal)

# p2 weight
def p2_weight(t_idx: torch.LongTensor):
    ab = alpha_bars.gather(0, t_idx).clamp(1e-8, 1-1e-8)
    snr = ab / (1.0 - ab)
    return (P2_K + snr) ** (-P2_GAMMA)

# integerize preserving sum
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
# DATASET (Iterable)
# -----------------------
class ShardedIterable(IterableDataset):
    """
    Ritorna:
      z0 (logit forma) [1,44,44] float32
      cond7_n [7] float32
      Y_real_sum [1] float32
      Y_real_map [1,44,44] float32 (per loss fisiche/metriche)
    """
    def __init__(self, shards, cond7_mean, cond7_std, shuffle_buffer=10000, seed=SEED):
        super().__init__()
        self.shards = list(shards)
        self.cond7_mean = cond7_mean.astype(np.float32)
        self.cond7_std  = np.maximum(cond7_std.astype(np.float32), COND_STD_FLOOR)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = seed

    def _iter_worker(self, shards_subset):
        rng = np.random.RandomState(self.seed + (get_worker_info().id if get_worker_info() else 0))
        buffer = []
        for sh in shards_subset:
            df = pd.read_pickle(sh)
            cond7_cols, psum_col, image_col = infer_columns(df)
            cond7 = df[cond7_cols].to_numpy(dtype=np.float32, copy=False)
            psum  = df[psum_col].to_numpy(dtype=np.float32, copy=False)
            imgs  = df[image_col].tolist()
            mats = [np.asarray(x, dtype=np.float32) for x in imgs]  # [N,44,44]

            cond7_n = (cond7 - self.cond7_mean[None,:]) / self.cond7_std[None,:]
            cond7_n = np.clip(cond7_n, -COND_CLIP, COND_CLIP)

            for i in range(len(mats)):
                Y = mats[i]
                s = float(Y.sum())
                if not np.isfinite(s) or s <= 0.0:
                    continue  # skip immagini vuote/degenerate
                z = to_shape_logits(Y, eps=EPS_P, z_clamp=Z_CLAMP)
                if z is None: continue
                item = (
                    torch.from_numpy(z).to(torch.float32),                             # [1,44,44]
                    torch.from_numpy(cond7_n[i]).view(COND7_DIM).to(torch.float32),   # [7]
                    torch.tensor([s], dtype=torch.float32),                            # [1]
                    torch.from_numpy(Y).unsqueeze(0).to(torch.float32)                # [1,44,44]
                )
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer:
                    j = rng.randint(0, len(buffer))
                    yield buffer.pop(j)

            rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()
            del df, cond7, psum, imgs, mats, cond7_n

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            order = list(self.shards); random.Random(self.seed).shuffle(order)
            return self._iter_worker(order)
        else:
            n = info.num_workers; wid = info.id
            order = list(self.shards); random.Random(self.seed + wid).shuffle(order)
            subset = [order[i] for i in range(len(order)) if i % n == wid]
            return self._iter_worker(subset)

# -----------------------
# SAMPLER (DDIM + CFG) su z-space
# -----------------------
@torch.no_grad()
def ddim_sample_cfg_vpred(unet_model, cond_vec, steps=8, cfg_scale=0.0):
    base = unwrap(unet_model); base.eval()
    B = cond_vec.size(0); H = W = IMG_SIZE
    x = torch.randn(B,1,H,W, device=DEVICE).contiguous(memory_format=torch.channels_last)
    seq = torch.linspace(T_STEPS-1, 0, steps, device=DEVICE, dtype=torch.long)
    zeros_cond = torch.zeros_like(cond_vec)

    for j, t in enumerate(seq):
        t = t.long()
        t_prev = seq[j+1].long() if j+1 < len(seq) else None
        sqrt_ab_t, sqrt_1mab_t, ab_t = gather_ab(t.expand(B))
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(torch.cuda.is_available() and not args.no_amp)):
            v_u = base(x, t.expand(B), zeros_cond)
            v_c = base(x, t.expand(B), cond_vec)
            v   = v_u + cfg_scale * (v_c - v_u)

        # x0 (== z0) e eps da v (v-pred identities)
        # x_t = c0 x0 + c1 eps ; v = c0 eps - c1 x0
        c0 = sqrt_ab_t.to(x.dtype); c1 = sqrt_1mab_t.to(x.dtype)
        z0 = c0 * x - c1 * v                      # x0 = c0 x_t - c1 v
        z0 = z0 - z0.flatten(1).mean(dim=1, keepdim=True).view(B,1,1,1)  # re-center
        z0 = torch.clamp(z0, min=-Z_CLAMP, max=Z_CLAMP)

        if t_prev is None:
            x = z0
            break

        # eps = c1 x_t + c0 v
        eps = c1 * x + c0 * v
        _, _, ab_prev = gather_ab(t_prev.expand(B))
        sqrt_ab_prev   = torch.sqrt(ab_prev).to(x.dtype)
        sqrt_1mab_prev = torch.sqrt(1.0 - ab_prev).to(x.dtype)
        x = sqrt_ab_prev * z0 + sqrt_1mab_prev * eps
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x  # z0

# -----------------------
# SETUP
# -----------------------
all_shards = list_train_shards(DATA_DIR)
train_shards = all_shards[:args.train_shards] if args.train_shards and args.train_shards>0 else all_shards
print(f"[DATA] Shard di train: {len(train_shards)} / tot={len(all_shards)}")

# Stats + BLR sufficient stats
cond7_mean, cond7_std, XtX, Xty, yTy, Ndata, y_min, y_max = cond7_and_blr_stats(train_shards, seed=SEED)
print(f"[STATS] cond7_mean shape={cond7_mean.shape} | N={Ndata} | y(min,max)=[{y_min:.3f},{y_max:.3f}]")

# BLR sampler oggetto (usa le stesse normalizzazioni del train)
blr_sampler = BLRScaleSampler(cond7_mean, cond7_std, XtX, Xty, yTy, Ndata, y_min, y_max)

# Dataloader
train_ds = ShardedIterable(train_shards, cond7_mean, cond7_std, shuffle_buffer=args.shuffle_buffer, seed=SEED)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=args.num_workers,
    pin_memory=True,
    prefetch_factor=args.prefetch_factor,
    persistent_workers=(args.num_workers > 0),
    drop_last=True
)

# Test set
test_df = load_test_df(DATA_DIR)
cond7_cols_te, psum_col_te, image_col_te = infer_columns(test_df)
cond7_te_np = test_df[cond7_cols_te].to_numpy(dtype=np.float32, copy=False)
imgs_te     = [np.asarray(x, dtype=np.float32) for x in test_df[image_col_te].tolist()]
mats_te_np  = np.stack(imgs_te, axis=0)  # float32
s_te_np     = mats_te_np.reshape(len(mats_te_np), -1).sum(axis=1).astype(np.float32)
mask_pos    = (s_te_np > 0)
mats_te_np  = mats_te_np[mask_pos]
cond7_te_np = cond7_te_np[mask_pos]
s_te_np     = s_te_np[mask_pos]

cond7_te_n  = (cond7_te_np - cond7_mean[None,:]) / np.maximum(cond7_std[None,:], COND_STD_FLOOR)
cond7_te_n  = np.clip(cond7_te_n, -COND_CLIP, COND_CLIP)
Z_te_list   = []
for i in range(mats_te_np.shape[0]):
    z = to_shape_logits(mats_te_np[i], eps=EPS_P, z_clamp=Z_CLAMP)
    Z_te_list.append(z)
Z_te = np.stack(Z_te_list, axis=0).astype(np.float32)  # [N,1,44,44]

cond7_te_t_all = torch.from_numpy(cond7_te_n).to(DEVICE)
Zte_all        = torch.from_numpy(Z_te).to(DEVICE).contiguous(memory_format=torch.channels_last)
Yte_real_all   = torch.from_numpy(mats_te_np).unsqueeze(1).to(DEVICE)
Ste_real_all   = torch.from_numpy(s_te_np).to(DEVICE)

# -----------------------
# MODEL/OPT/EMA
# -----------------------
unet = CondUNetV(cond_dim=COND7_DIM, time_dim=TIME_DIM, cond_emb=COND_EMB, base_ch=BASE_CH).to(DEVICE)
unet = unet.to(memory_format=torch.channels_last)

if not args.no_compile:
    try:
        unet = torch.compile(unet, mode="reduce-overhead", fullgraph=False)
    except Exception as e:
        print(f"[WARN] torch.compile disabled ({e})")

ema_unet = EMA(unet, decay=EMA_DECAY)
opt = optim.AdamW(unet.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
mse = nn.MSELoss(reduction='none')
SCALER = torch.amp.GradScaler('cuda', enabled=(torch.cuda.is_available() and not args.no_amp))

# -----------------------
# TRAIN / LOSS
# -----------------------
def train_epoch(epoch_idx=None):
    unet.train()
    total = 0.0; nb = 0
    total_v = total_center = total_peak = total_rad = 0.0

    for (z0_cpu, cond7_cpu, s_cpu, Y_cpu) in train_loader:
        z0 = z0_cpu.to(DEVICE, non_blocking=True)              # [B,1,H,W]
        cond7 = cond7_cpu.to(DEVICE, non_blocking=True)        # [B,7]
        s_real = s_cpu.to(DEVICE, non_blocking=True).view(-1)  # [B]
        Y_real = Y_cpu.to(DEVICE, non_blocking=True)           # [B,1,H,W]
        Bcur = z0.size(0)

        # classifier-free dropout sul cond
        use_uncond = (torch.rand(Bcur, device=DEVICE) < CFG_P_UNCOND).float().view(-1,1)
        zeros = torch.zeros_like(cond7)
        cond_vec = use_uncond * zeros + (1.0 - use_uncond) * cond7

        t = torch.randint(0, T_STEPS, (Bcur,), device=DEVICE, dtype=torch.long)
        sqrt_ab_t, sqrt_1mab_t, _ = gather_ab(t)
        eps = torch.randn_like(z0)
        x_t = sqrt_ab_t.to(z0.dtype) * z0 + sqrt_1mab_t.to(z0.dtype) * eps

        # target v = c0*eps - c1*z0
        c0 = sqrt_ab_t.to(z0.dtype); c1 = sqrt_1mab_t.to(z0.dtype)
        v_tgt = c0 * eps - c1 * z0

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(torch.cuda.is_available() and not args.no_amp)):
            v_pred = unet(x_t, t, cond_vec)
            per_pix = (v_pred - v_tgt)**2
            per_img = per_pix.mean(dim=(1,2,3))
            w_t = p2_weight(t)
            loss_v = (per_img * w_t).mean()

            # ricostruisci z0 -> p -> Y_hat (usando s reale per data-consistency)
            z0_hat = c0 * x_t - c1 * v_pred
            # re-center/clamp prima della softmax
            z0_hat = z0_hat - z0_hat.flatten(1).mean(dim=1, keepdim=True).view(Bcur,1,1,1)
            z0_hat = torch.clamp(z0_hat, min=-Z_CLAMP, max=Z_CLAMP)
            p_hat = softmax_shape(z0_hat)                       # [B,1,H,W]
            Y_hat = p_hat * s_real.view(-1,1,1,1)               # [B,1,H,W]

            # loss fisiche
            # center
            cx_r_u, cy_r_u = soft_argmax_xy_unit_stable(Y_real, tau=CENTER_TAU)
            cx_p_u, cy_p_u = soft_argmax_xy_unit_stable(Y_hat,  tau=CENTER_TAU)
            H=W=IMG_SIZE
            cx_r_px = cx_r_u * (W-1); cy_r_px = cy_r_u * (H-1)
            cx_p_px = cx_p_u * (W-1); cy_p_px = cy_p_u * (H-1)
            loss_center = (cx_r_px - cx_p_px).abs().mean() + (cy_r_px - cy_p_px).abs().mean()

            # peak log
            gmax_r = Y_real.flatten(1).max(dim=1).values
            gmax_p = Y_hat .flatten(1).max(dim=1).values
            def huber(a,b,delta=0.5):
                diff = a-b; ad = diff.abs()
                quad = torch.clamp(ad, max=delta)
                lin = ad - quad
                return 0.5*(quad**2) + delta*lin
            loss_peak = huber(torch.log1p(gmax_p), torch.log1p(gmax_r), delta=0.5).mean()

            # radial CDF (su downsample x2)
            loss_rad = radial_cdf_loss(down2(Y_hat), down2(Y_real), cx_p_px/2.0, cy_p_px/2.0,
                                       radii=RADII_PX, delta=RADIAL_SIGMOID_DELTA)

            loss = loss_v + W_CENTER_PX*loss_center + W_PEAK_LOG*loss_peak + W_SHAPE_RAD*loss_rad

        if not torch.isfinite(loss):
            continue

        if SCALER.is_enabled():
            SCALER.scale(loss).backward()
            if GRAD_CLIP:
                SCALER.unscale_(opt)
                nn.utils.clip_grad_norm_(unet.parameters(), GRAD_CLIP)
            SCALER.step(opt); SCALER.update()
        else:
            loss.backward()
            if GRAD_CLIP: nn.utils.clip_grad_norm_(unet.parameters(), GRAD_CLIP)
            opt.step()

        ema_unet.update(unet)
        bs = Bcur
        total += float(loss.detach())*bs; nb += bs
        total_v += float(loss_v.detach())*bs
        total_center += float(loss_center.detach())*bs
        total_peak += float(loss_peak.detach())*bs
        total_rad += float(loss_rad.detach())*bs

    nb = max(1, nb)
    logs = dict(
        loss_total=total/nb, loss_v=total_v/nb, loss_center=total_center/nb,
        loss_peak=total_peak/nb, loss_rad=total_rad/nb
    )
    # CSV loss components
    file_exists= os.path.isfile("loss_component_all.csv")
    with open("loss_component_all.csv", mode="a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["epoch","loss_total","loss_v","loss_center","loss_peak","loss_rad"])
        w.writerow([int(epoch_idx) if epoch_idx is not None else ""] + [logs[k] for k in ["loss_total","loss_v","loss_center","loss_peak","loss_rad"]])
    return logs["loss_total"]

# -----------------------
# EVAL
# -----------------------
@torch.no_grad()
def eval_end_to_end(ema_u_state, split="test", num=EVAL_NUM, steps=EVAL_STEPS, cfg_scale=CFG_SCALE_EVAL,
                    int_eval=False, eval_debug=False):
    # ricreo modello EMA
    u = CondUNetV(cond_dim=COND7_DIM, time_dim=TIME_DIM, cond_emb=COND_EMB, base_ch=BASE_CH).to(DEVICE)
    u = u.to(memory_format=torch.channels_last)
    u.load_state_dict(ema_u_state, strict=True); u.eval()

    idx = torch.arange(min(num, Zte_all.size(0)), device=DEVICE)
    cond7 = cond7_te_t_all[idx]
    z_real = Zte_all[idx]
    Y_real = Yte_real_all[idx]
    s_real = Ste_real_all[idx]

    # sampling in z-space (forma)
    z0 = ddim_sample_cfg_vpred(u, cond7.to(torch.float32), steps=steps, cfg_scale=cfg_scale)
    p  = softmax_shape(z0)
    # (A) metriche shape-only: usa s reale
    Y_hat_shape = p * s_real.view(-1,1,1,1)

    # (B) metriche full: campiona s da BLR predittiva
    s_hat = blr_sampler.sample_s(cond7.to(torch.float32), clip_quantiles=tuple(args.blr_clip_quantiles))
    Y_hat_full = p * s_hat.view(-1,1,1,1)

    # metriche
    def mse_rmse_sse(A, B):
        diff2 = (A - B) ** 2
        msepp = diff2.mean().item()
        rmse  = float(msepp ** 0.5)
        sse   = diff2.sum().item()
        return msepp, rmse, sse

    mseA, rmseA, sseA = mse_rmse_sse(Y_hat_shape, Y_real)
    mseB, rmseB, sseB = mse_rmse_sse(Y_hat_full,  Y_real)

    if int_eval:
        Y_int_A = integerize_with_sum(Y_hat_shape, torch.round(s_real).clamp(min=0).to(torch.long))
        Y_int_B = integerize_with_sum(Y_hat_full,  torch.round(s_hat).clamp(min=0).to(torch.long))
        d2A = (Y_int_A.float() - Y_real.float())**2
        d2B = (Y_int_B.float() - Y_real.float())**2
        mseA = d2A.mean().item(); rmseA = float(mseA**0.5); sseA = d2A.sum().item()
        mseB = d2B.mean().item(); rmseB = float(mseB**0.5); sseB = d2B.sum().item()

    # sparsity diag
    thr = 1.0
    z_real_frac = (Y_real <= thr).float().mean().item()
    z_fakeA_frac = (Y_hat_shape <= thr).float().mean().item()
    z_fakeB_frac = (Y_hat_full  <= thr).float().mean().item()

    # sum error diag
    sum_real = s_real
    sum_A = Y_hat_shape.flatten(1).sum(dim=1)
    sum_B = Y_hat_full .flatten(1).sum(dim=1)
    mean_abs_sum_err_A = (sum_A - sum_real).abs().mean().item()
    mean_abs_sum_err_B = (sum_B - sum_real).abs().mean().item()

    if eval_debug:
        print(f"[EVAL DEBUG] mean(sum_real)={float(sum_real.mean()):.2f} | "
              f"mean(sum_hat_full)={float(sum_B.mean()):.2f} | "
              f"mean(max_real)={float(Y_real.flatten(1).max(dim=1).values.mean()):.2f} | "
              f"mean(max_full)={float(Y_hat_full.flatten(1).max(dim=1).values.mean()):.2f}")

    return (mseA, rmseA, sseA, z_real_frac, z_fakeA_frac, mean_abs_sum_err_A,
            mseB, rmseB, sseB, z_fakeB_frac, mean_abs_sum_err_B)

# -----------------------
# TRAIN LOOP
# -----------------------
best_mse_shape = float("inf")
best_mse_full  = float("inf")
P = IMG_SIZE * IMG_SIZE

for ep in range(1, EPOCHS+1):
    t0 = time.time()
    print(f"[EPOCH {ep}] loader ready | workers={args.num_workers} | batch={BATCH_SIZE} | prefetch={args.prefetch_factor}")
    tr_loss = train_epoch(ep)

    ema_u = {k: v.clone() for k, v in ema_unet.shadow.items()}

    (mseA, rmseA, sseA, z_real_t, z_fakeA_t, sum_err_A,
     mseB, rmseB, sseB, z_fakeB_t, sum_err_B) = eval_end_to_end(
        ema_u, split="test", int_eval=args.int_eval, steps=EVAL_STEPS, cfg_scale=CFG_SCALE_EVAL, eval_debug=args.eval_debug)

    dt = time.time() - t0
    print(f"[Epoch {ep:03d}] loss={tr_loss:.5f} | "
          f"SHAPE: MSEpp={mseA:.4f} RMSE={rmseA:.4f} SSE={sseA:.1f} ≈ {mseA:.2f}*{P} zeros(real)={z_real_t:.3f} zeros(fake)={z_fakeA_t:.3f} Δpsum|mean|={sum_err_A:.3f} | "
          f"FULL:  MSEpp={mseB:.4f} RMSE={rmseB:.4f} SSE={sseB:.1f} zeros(fake)={z_fakeB_t:.3f} Δpsum|mean|={sum_err_B:.3f} | "
          f"{dt:.1f}s")

    ckpt_dir = Path("DDPM_shape_blr_ckpt"); ckpt_dir.mkdir(exist_ok=True)
    epoch_ckpt = {
        "epoch": ep,
        "unet_ema": ema_u,
        # cond7 stats
        "cond7_mean": torch.from_numpy(cond7_mean).float(),
        "cond7_std":  torch.from_numpy(cond7_std).float(),
        # BLR stats
        "blr": {
            "XtX": torch.from_numpy(XtX).double(),
            "Xty": torch.from_numpy(Xty).double(),
            "yTy": float(yTy),
            "N":   int(Ndata),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        # cfg
        "cfg": {"IMG_SIZE": IMG_SIZE, "COND7_DIM": COND7_DIM, "TIME_DIM": TIME_DIM, "COND_EMB": COND_EMB,
                "BASE_CH": BASE_CH, "T_STEPS": T_STEPS}
    }
    torch.save(epoch_ckpt, ckpt_dir / f"epoch_{ep:03d}.pt")

    improved = False
    if mseA < best_mse_shape and np.isfinite(mseA):
        best_mse_shape = mseA
        torch.save(epoch_ckpt, "best_shape_checkpoint.pt"); improved = True
    if mseB < best_mse_full and np.isfinite(mseB):
        best_mse_full = mseB
        torch.save(epoch_ckpt, "best_full_checkpoint.pt"); improved = True
    if improved:
        print(f"  -> salvati best_*_checkpoint.pt (shape={best_mse_shape:.4f}, full={best_mse_full:.4f})")

    # CSV metrics
    metrics_path = Path("DDPM_shape_blr_metrics.csv")
    log_row = {
        "epoch": ep,
        "train_total_loss": float(tr_loss),
        "shape_mse_pp": float(mseA),
        "shape_rmse": float(rmseA),
        "shape_sse": float(sseA),
        "full_mse_pp": float(mseB),
        "full_rmse": float(rmseB),
        "full_sse": float(sseB),
        "zeros_real": float(z_real_t),
        "zeros_fake_shape": float(z_fakeA_t),
        "zeros_fake_full": float(z_fakeB_t),
        "mean_abs_sum_err_shape": float(sum_err_A),
        "mean_abs_sum_err_full": float(sum_err_B),
        "lr": float(opt.param_groups[0]["lr"]),
        "eval_steps": int(EVAL_STEPS),
        "cfg_scale_eval": float(CFG_SCALE_EVAL),
        "p_uncond": float(CFG_P_UNCOND),
        "epoch_time_sec": float(dt),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "shuffle_buffer": int(args.shuffle_buffer),
        "train_shards_used": int(len(train_shards)),
        "batch_size": int(BATCH_SIZE),
    }
    pd.DataFrame([log_row]).to_csv(
        metrics_path, mode="a", header=not metrics_path.exists(), index=False
    )

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print("Training finito.")

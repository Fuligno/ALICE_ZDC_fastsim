# DDPM_infer_dists.py
# - Visualizza 4x2 (REAL vs GEN)
# - Confronta distribuzioni (entropia normalizzata, somma, massimo) su un subsample del test
# - Generazione robusta in chunk (+ opzionale fp16) per evitare OOM
import argparse, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------
# Args
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="/data/dataalice/dfuligno/tesi/myTesi/Dati_processed")
parser.add_argument("--ckpt", type=str, default="best_full_checkpoint.pt")
parser.add_argument("--eval-steps", type=int, default=8)
parser.add_argument("--cfg-scale", type=float, default=0.5)
parser.add_argument("--num", type=int, default=4, help="quante coppie (righe) plottare")
parser.add_argument("--dist-n", type=int, default=2048, help="N campioni test per stimare le distribuzioni")
parser.add_argument("--chunk", type=int, default=128, help="chunk size per generazione distribuzioni")
parser.add_argument("--fp16", action="store_true", help="usa half precision in inferenza (CUDA)")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
DEVICE = torch.device(args.device)

# -----------------------
# Utils coerenti col training
# -----------------------
IMG_SIZE = 44
COND7_DIM = 7
T_STEPS = 1000
Z_CLAMP = 8.0
COND_STD_FLOOR = 1e-2
COND_CLIP = 8.0
EPS = 1e-12

def cosine_alpha_bars(T: int, s: float = 0.008, device=DEVICE, dtype=torch.float32):
    steps = torch.arange(T + 1, dtype=dtype, device=device)
    t = steps / T
    a_bars_full = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a_bars_full = a_bars_full / a_bars_full[0]
    alpha_bars = a_bars_full[1:]
    alphas = alpha_bars / a_bars_full[:-1]
    betas = 1.0 - alphas
    return betas.clamp(1e-8, 0.999), alphas, alpha_bars

_, _, ALPHA_BARS = cosine_alpha_bars(T_STEPS, device=DEVICE, dtype=torch.float32)

def gather_ab(t_idx: torch.LongTensor):
    ab = ALPHA_BARS.gather(0, t_idx).view(-1,1,1,1)
    return torch.sqrt(ab), torch.sqrt(1.0 - ab), ab

def softmax_shape(z0: torch.Tensor):
    B,_,H,W = z0.shape
    z = z0 - z0.flatten(1).mean(dim=1, keepdim=True).view(B,1,1,1)
    z = torch.clamp(z, min=-Z_CLAMP, max=Z_CLAMP)
    p = torch.softmax(z.flatten(1), dim=1).view(B,1,H,W)
    return p

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
            if cand in cols:
                psum_col = cand; break
        else:
            psum_col = cols[8]
        cond7_cols = cols[1:8]
    else:
        cond7_cols = cols[1:8]; psum_col = cols[8]; image_col = cols[9]
    return cond7_cols, psum_col, image_col

def entropy_normalized(y: np.ndarray) -> float:
    s = float(y.sum())
    if s <= 0:
        return 0.0
    p = (y.reshape(-1) / s).astype(np.float64)
    p = np.clip(p, EPS, 1.0)
    H = -np.sum(p * np.log(p))
    Hmax = math.log(y.size)
    return float(H / Hmax) if Hmax > 0 else 0.0

# -----------------------
# Modello (come training)
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
    def __init__(self, cond_dim=7, time_dim=64, cond_emb=64, base_ch=48):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim,cond_emb), nn.SiLU(), nn.Linear(cond_emb,cond_emb))
        emb_dim = time_dim + cond_emb
        ch = base_ch
        self.stem = nn.Conv2d(3, ch, 3, padding=1)
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
        return self.out(x)

# v-DDIM sampler (z-space) compatibile fp16
@torch.no_grad()
def ddim_sample_cfg_vpred(unet_model, cond_vec, steps=8, cfg_scale=0.0):
    base = unet_model.eval()
    B = cond_vec.size(0); H = W = IMG_SIZE
    use_fp16 = (DEVICE.type == "cuda") and (next(base.parameters()).dtype == torch.float16)
    dtype = torch.float16 if use_fp16 else torch.float32

    x = torch.randn(B,1,H,W, device=DEVICE, dtype=dtype).contiguous(memory_format=torch.channels_last)
    seq = torch.linspace(T_STEPS-1, 0, steps, device=DEVICE, dtype=torch.long)
    zeros_cond = torch.zeros_like(cond_vec)

    use_autocast = (DEVICE.type == "cuda") and use_fp16
    autocast_ctx = torch.cuda.amp.autocast if use_autocast else torch.autocast  # torch.autocast will be disabled if device='cpu'

    with (torch.cuda.amp.autocast(enabled=True) if use_autocast else torch.no_grad()):
        for j, t in enumerate(seq):
            t = t.long()
            t_prev = seq[j+1].long() if j+1 < len(seq) else None
            sqrt_ab_t, sqrt_1mab_t, _ = gather_ab(t.expand(B))
            v_u = base(x, t.expand(B), zeros_cond)
            v_c = base(x, t.expand(B), cond_vec)
            v   = v_u + cfg_scale * (v_c - v_u)
            c0 = sqrt_ab_t.to(x.dtype); c1 = sqrt_1mab_t.to(x.dtype)
            z0 = c0 * x - c1 * v
            z0 = z0 - z0.flatten(1).mean(dim=1, keepdim=True).view(B,1,1,1)
            z0 = torch.clamp(z0, min=-Z_CLAMP, max=Z_CLAMP)
            if t_prev is None:
                x = z0
                break
            eps = c1 * x + c0 * v
            _, _, ab_prev = gather_ab(t_prev.expand(B))
            sqrt_ab_prev   = torch.sqrt(ab_prev).to(x.dtype)
            sqrt_1mab_prev = torch.sqrt(1.0 - ab_prev).to(x.dtype)
            x = sqrt_ab_prev * z0 + sqrt_1mab_prev * eps
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

# -----------------------
# BLR Sampler (posterior predittiva t-Student)
# -----------------------
class BLRScaleSampler:
    def __init__(self, mean_c, std_c, XtX, Xty, yTy, N, y_min, y_max):
        self.mean_c = torch.tensor(mean_c, dtype=torch.float32)
        self.std_c  = torch.tensor(np.maximum(std_c, COND_STD_FLOOR), dtype=torch.float32)
        self.XtX = torch.tensor(XtX, dtype=torch.float64)
        self.Xty = torch.tensor(Xty, dtype=torch.float64)
        self.yTy = float(yTy); self.N = int(N); self.p = self.XtX.shape[0]
        self.y_min = float(y_min); self.y_max = float(y_max)
        self.XtX_inv = torch.linalg.pinv(self.XtX)
        self.beta = (self.XtX_inv @ self.Xty)
        beta_col = self.beta.view(-1,1)
        SSE = self.yTy - 2.0*float((self.beta @ self.Xty).item()) + float((beta_col.T @ (self.XtX @ beta_col)).item())
        dof = max(1, self.N - self.p)
        self.sigma2 = max(1e-8, SSE / dof)
        self.dof = dof

    @torch.no_grad()
    def sample_s(self, cond7: torch.Tensor, clip_quantiles=(0.005, 0.995)) -> torch.Tensor:
        device = cond7.device
        c_n = (cond7.float() - self.mean_c.to(device)) / self.std_c.to(device)
        c_n = torch.clamp(c_n, -COND_CLIP, COND_CLIP)
        ones = torch.ones(c_n.size(0), 1, device=device, dtype=torch.float32)
        X = torch.cat([ones, c_n], dim=1).to(torch.float64)
        beta     = self.beta.to(device)
        XtX_inv  = self.XtX_inv.to(device)
        sigma2   = torch.as_tensor(self.sigma2, dtype=torch.float64, device=device)
        m = (X @ beta)
        X_V = X @ XtX_inv
        h = (X_V * X).sum(dim=1)
        v = sigma2 * (1.0 + h)
        v = torch.clamp(v, min=torch.finfo(v.dtype).eps)
        loc = m.to(torch.float32); scale = torch.sqrt(v).to(torch.float32)
        dist = torch.distributions.StudentT(df=float(self.dof), loc=loc, scale=scale)
        y = dist.sample()
        q_low, q_high = clip_quantiles
        y_lo = torch.tensor(self.y_min + (self.y_max-self.y_min)*q_low, device=device, dtype=y.dtype)
        y_hi = torch.tensor(self.y_min + (self.y_max-self.y_min)*q_high, device=device, dtype=y.dtype)
        y = torch.clamp(y, min=y_lo, max=y_hi)
        s = torch.expm1(y)
        return torch.clamp(s, min=0.0)

# -----------------------
# Load checkpoint + rebuild
# -----------------------
ckpt = torch.load(args.ckpt, map_location="cpu")
cfg = ckpt.get("cfg", {})
BASE_CH = cfg.get("BASE_CH", 48)
TIME_DIM = cfg.get("TIME_DIM", 64)

cond7_mean = ckpt["cond7_mean"].float().numpy()
cond7_std  = ckpt["cond7_std"].float().numpy()

blr_stats = ckpt["blr"]
blr_sampler = BLRScaleSampler(
    cond7_mean, cond7_std,
    blr_stats["XtX"].numpy(), blr_stats["Xty"].numpy(),
    blr_stats["yTy"], blr_stats["N"],
    blr_stats["y_min"], blr_stats["y_max"]
)

unet = CondUNetV(cond_dim=COND7_DIM, time_dim=TIME_DIM, cond_emb=64, base_ch=BASE_CH).to(DEVICE).to(memory_format=torch.channels_last)
unet.load_state_dict(ckpt["unet_ema"])
if args.fp16 and DEVICE.type == "cuda":
    unet.half()
unet.eval()

# -----------------------
# Carica test e seleziona validi
# -----------------------
df_test = load_test_df(args.data_dir)
cond7_cols, psum_col, image_col = infer_columns(df_test)
cond7_np = df_test[cond7_cols].to_numpy(dtype=np.float32, copy=False)
imgs = [np.asarray(x, dtype=np.float32) for x in df_test[image_col].tolist()]
mats = np.stack(imgs, axis=0)  # [N,44,44]
sums = mats.reshape(len(mats), -1).sum(axis=1).astype(np.float32)
mask = sums > 0
cond7_np = cond7_np[mask]
mats = mats[mask]
sums = sums[mask]
N = cond7_np.shape[0]

# -----------------------
# Helpers
# -----------------------
def normalize_cond7(c):
    c_n = (c - cond7_mean[None,:]) / np.maximum(cond7_std[None,:], COND_STD_FLOOR)
    return np.clip(c_n, -COND_CLIP, COND_CLIP)

@torch.no_grad()
def generate_in_chunks(unet_model, cond7t, steps, cfg_scale, chunk):
    outs = []
    use_fp16 = (DEVICE.type == "cuda") and (next(unet_model.parameters()).dtype == torch.float16)
    for i in range(0, cond7t.size(0), chunk):
        sl = cond7t[i:i+chunk]
        sl = sl.to(torch.float16 if use_fp16 else torch.float32)
        z0 = ddim_sample_cfg_vpred(unet_model, sl, steps=steps, cfg_scale=cfg_scale)
        p  = softmax_shape(z0)  # [b,1,44,44]
        s_hat_b = blr_sampler.sample_s(sl)  # [b]
        Y_gen_b = (p * s_hat_b.view(-1,1,1,1)).squeeze(1).detach().cpu()
        outs.append(Y_gen_b)
        del z0, p, s_hat_b, Y_gen_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(outs, dim=0).numpy()

# -----------------------
# 1) VIZ 4x2 immagini
# -----------------------
idxs_viz = np.random.choice(N, size=min(args.num, N), replace=False)
cond7_sel = cond7_np[idxs_viz]
Y_real_sel = mats[idxs_viz]

cond7_n = normalize_cond7(cond7_sel)
cond7_t = torch.from_numpy(cond7_n).to(DEVICE)

Y_gen_viz = generate_in_chunks(unet, cond7_t, steps=args.eval_steps, cfg_scale=args.cfg_scale, chunk=max(1, min(args.chunk, cond7_t.size(0))))

Y_real_t = torch.from_numpy(Y_real_sel).unsqueeze(1).to(DEVICE)
Y_gen_t  = torch.from_numpy(Y_gen_viz).unsqueeze(1).to(DEVICE)

# Plot 4x2
B = Y_gen_t.size(0)
rows = B
fig, axes = plt.subplots(rows, 2, figsize=(8, 2.2*rows), constrained_layout=True)
if rows == 1: axes = np.array([axes])

for i in range(rows):
    y_real = Y_real_t[i,0].detach().cpu().numpy()
    y_gen  = Y_gen_t[i,0].detach().cpu().numpy()

    axL = axes[i,0]
    imL = axL.imshow(y_real, origin="lower", aspect="equal")
    cbarL = fig.colorbar(imL, ax=axL, fraction=0.046, pad=0.04)
    cbarL.set_label("counts", rotation=270, labelpad=10)
    axL.set_title(f"REAL  | sum={y_real.sum():.1f}, max={y_real.max():.1f}")
    axL.set_xticks([]); axL.set_yticks([])

    axR = axes[i,1]
    imR = axR.imshow(y_gen, origin="lower", aspect="equal")
    cbarR = fig.colorbar(imR, ax=axR, fraction=0.046, pad=0.04)
    cbarR.set_label("counts", rotation=270, labelpad=10)
    axR.set_title(f"GEN   | sum={y_gen.sum():.1f}, max={y_gen.max():.1f}")
    axR.set_xticks([]); axR.set_yticks([])

fig.suptitle("Calorimeter response — REAL vs GENERATED (independent z-scale per image)", fontsize=12)
plt.savefig("confronto_4x2.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -----------------------
# 2) DISTRIBUZIONI su subsample condiviso
# -----------------------
M = min(args.dist_n, N)
idxs_dist = np.random.choice(N, size=M, replace=False)

Y_real_dist = mats[idxs_dist].astype(np.float64)
cond7_dist = cond7_np[idxs_dist]
cond7n_dist = normalize_cond7(cond7_dist)
cond7t_dist = torch.from_numpy(cond7n_dist).to(DEVICE)

Y_gen_dist = generate_in_chunks(unet, cond7t_dist, steps=args.eval_steps, cfg_scale=args.cfg_scale, chunk=args.chunk)

sum_real = Y_real_dist.reshape(M, -1).sum(axis=1)
sum_gen  = Y_gen_dist.reshape(M, -1).sum(axis=1)

max_real = Y_real_dist.reshape(M, -1).max(axis=1)
max_gen  = Y_gen_dist.reshape(M, -1).max(axis=1)

Hn_real = np.array([entropy_normalized(Y_real_dist[i]) for i in range(M)], dtype=np.float64)
Hn_gen  = np.array([entropy_normalized(Y_gen_dist[i])  for i in range(M)], dtype=np.float64)

# salva CSV
out_df = pd.DataFrame({
    "sum_real": sum_real,
    "sum_gen": sum_gen,
    "max_real": max_real,
    "max_gen": max_gen,
    "Hn_real": Hn_real,
    "Hn_gen": Hn_gen
})
out_df.to_csv("dist_metrics_sample.csv", index=False)

# -----------------------
# Plot istogrammi comparativi
# -----------------------
def _nice_bins(x_real, x_gen, nb=60, positive=False):
    x = np.concatenate([x_real, x_gen])
    lo, hi = np.nanpercentile(x, [0.5, 99.5])
    if positive: lo = max(0.0, lo)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x) + 1e-9)
    bins = np.linspace(lo, hi, nb+1)
    return bins, (lo, hi)

# Entropia normalizzata [0,1]
bins_H = np.linspace(0.0, 1.0, 51)
plt.figure(figsize=(7.2, 4.0))
plt.hist(Hn_real, bins=bins_H, alpha=0.55, density=True, label="REAL", edgecolor="black", linewidth=0.3)
plt.hist(Hn_gen,  bins=bins_H, alpha=0.55, density=True, label="GEN",  edgecolor="black", linewidth=0.3)
plt.xlabel("Normalized entropy H / log(H·W)"); plt.ylabel("Density")
plt.title(f"Entropy distribution (M={M})")
plt.legend(); plt.tight_layout(); plt.savefig("dist_entropy_norm.png", dpi=300); plt.close()

# Somma
bins_S, rng_S = _nice_bins(sum_real, sum_gen, nb=60, positive=True)
plt.figure(figsize=(7.2, 4.0))
plt.hist(sum_real, bins=bins_S, alpha=0.55, density=True, label="REAL", edgecolor="black", linewidth=0.3)
plt.hist(sum_gen,  bins=bins_S, alpha=0.55, density=True, label="GEN",  edgecolor="black", linewidth=0.3)
plt.xlabel("Matrix sum (counts)"); plt.ylabel("Density")
plt.title(f"Sum distribution (M={M})  range≈[{rng_S[0]:.1f},{rng_S[1]:.1f}]")
plt.legend(); plt.tight_layout(); plt.savefig("dist_sum.png", dpi=300); plt.close()

# Max
bins_M, rng_M = _nice_bins(max_real, max_gen, nb=60, positive=True)
plt.figure(figsize=(7.2, 4.0))
plt.hist(max_real, bins=bins_M, alpha=0.55, density=True, label="REAL", edgecolor="black", linewidth=0.3)
plt.hist(max_gen,  bins=bins_M, alpha=0.55, density=True, label="GEN",  edgecolor="black", linewidth=0.3)
plt.xlabel("Matrix max (counts)"); plt.ylabel("Density")
plt.title(f"Max distribution (M={M})  range≈[{rng_M[0]:.1f},{rng_M[1]:.1f}]")
plt.legend(); plt.tight_layout(); plt.savefig("dist_max.png", dpi=300); plt.close()

print("✓ Salvati: confronto_4x2.png, dist_entropy_norm.png, dist_sum.png, dist_max.png, dist_metrics_sample.csv")

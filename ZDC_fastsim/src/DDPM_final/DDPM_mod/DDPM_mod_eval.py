# ============================================================
# DDPM_infer_stats_streaming_with_spread_FAST.py
# - Streaming, RAM/VRAM costante
# - Patch: CFG concat (un forward), proiezione solo finale (o ogni K step)
# - Timing per batch: t_gen (sampling+decode), t_proc (metriche), t_tot
# - Distribuzioni 1D (γ_max, γ_tot, entropia) + istogrammi 2D spread e proiezioni
# ============================================================
import os, math, random, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------
# CONFIG (modifica qui)
# -----------------------
DATA_DIR     = "/data/dataalice/dfuligno/tesi/myTesi/Dati_processed"
CKPT_PATH    = "best_ema_checkpoint.pt"
BATCH_GEN    = 256                # alza finché la VRAM regge (con CFG-concat usa 2B in forward)
EVAL_STEPS   = 100                 # meno step = molto più veloce (50~30 spesso ok per distribuzioni)
CFG_SCALE    = 3.0
ETA          = 0.0
PROJ_EVERY_K = 0                  # 0 = proietta SOLO all'ultimo step; se >0, proietta ogni K step
SEED         = 42
SAVE_SAMPLE_K= 0                  # 0 = non salva esempi di debug
USE_LOG_IMSHOW = False            # True per heatmap in scala log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.float16
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except: pass

# -----------------------
# Utils I/O e dati
# -----------------------
def load_test_df(data_dir: str):
    p = Path(data_dir) / "test.pkl"
    if not p.is_file():
        alt = Path(data_dir) / "test.plk"
        if not alt.is_file():
            raise FileNotFoundError(f"Manca test.pkl/plk in {data_dir}")
        p = alt
    return pd.read_pickle(p)

def infer_columns(df: pd.DataFrame):
    cols = list(df.columns)
    if "image" in cols:
        image_col = "image"
        for cand in ["photonSum", "psum", "PhotonSum", "psum_total"]:
            if cand in cols:
                psum_col = cand
                break
        else:
            psum_col = cols[8]  # fallback
        cond7_cols = cols[1:8]
    else:
        cond7_cols = cols[1:8]; psum_col = cols[8]; image_col = cols[9]
    return cond7_cols, psum_col, image_col

# -----------------------
# Diffusion schedule
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

def gather_ab(t_idx: torch.LongTensor, alpha_bars: torch.Tensor):
    ab = alpha_bars.gather(0, t_idx).view(-1,1,1,1)
    return torch.sqrt(ab), torch.sqrt(1.0 - ab), ab

# -----------------------
# Modelli (come in training)
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
        self.skip = (in_ch != out_ch); self.conv_skip = nn.Conv2d(in_ch, out_ch, 1) if self.skip else None
    def forward(self, x, emb):
        scale, shift = self.emb(emb).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1); shift = shift.unsqueeze(-1).unsqueeze(-1)
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.norm2(h); h = h * (1 + scale.to(h.dtype)) + shift.to(h.dtype)
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

# -----------------------
# Decode & helpers
# -----------------------
MM1_CLAMP_EVAL = 1.0
NAN_REPLACE_VAL = 0.0

def decode_target_torch(y_mm1: torch.Tensor, y_log_scale: float):
    y_mm1 = y_mm1.clamp(min=y_mm1.new_tensor(-MM1_CLAMP_EVAL), max=y_mm1.new_tensor(MM1_CLAMP_EVAL))
    y_01  = (y_mm1 + y_mm1.new_tensor(1.0)) * y_mm1.new_tensor(0.5)
    y_log = y_01 * y_mm1.new_tensor(y_log_scale, dtype=y_mm1.dtype)
    out   = torch.expm1(y_log)
    return torch.nan_to_num(out, nan=NAN_REPLACE_VAL, posinf=1e6, neginf=0.0)

def project_positive_with_sum(x_real: torch.Tensor, sum_target: torch.Tensor, eps: float = 1e-6):
    x_pos = torch.relu(x_real)
    s = x_pos.flatten(1).sum(dim=1, keepdim=True)
    eps_t = torch.as_tensor(eps, device=x_real.device, dtype=x_real.dtype)
    scale = (sum_target.view(-1,1).to(x_real.dtype) / (s + eps_t)).clamp(min=x_real.new_tensor(0.0), max=x_real.new_tensor(1e6))
    out = x_pos * scale.view(-1,1,1,1)
    return torch.nan_to_num(out, nan=NAN_REPLACE_VAL, posinf=1e6, neginf=0.0)

# -----------------------
# DDIM SAMPLER — FAST (CFG concat + proiezione finale / ogni K)
# -----------------------
@torch.inference_mode()
def ddim_sample_cfg_fast(unet_model, cond_vec, x_u, y_u, psum_pred_real, alpha_bars, T_STEPS, IMG_SIZE, PRIOR_SIGMA_PX,
                         y_log_scale, y_clip_max, steps=50, cfg_scale=3.0, eta=0.0, proj_every_k: int = 0):
    """
    proj_every_k: 0 = proietta solo all'ultimo step; >0 = proietta anche ogni K step
    """
    base = unet_model.eval()
    B = cond_vec.size(0); H = W = IMG_SIZE
    x = torch.randn(B,1,H,W, device=DEVICE).contiguous(memory_format=torch.channels_last)
    seq = torch.linspace(T_STEPS-1, 0, steps, device=DEVICE, dtype=torch.long)
    zeros_cond = torch.zeros_like(cond_vec)
    prior = gaussian_prior(x_u, y_u, H, W, PRIOR_SIGMA_PX, DEVICE, x.dtype)

    for j, t in enumerate(seq):
        t = t.long()
        t_prev = seq[j+1].long() if j+1 < len(seq) else None
        sqrt_ab_t, sqrt_1mab_t, ab_t = gather_ab(t.expand(B), alpha_bars)

        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            # ---- CFG concat: un solo forward su batch 2B ----
            tB    = t.expand(B)
            x_cat = torch.cat([x, x], dim=0)
            c_cat = torch.cat([zeros_cond, cond_vec], dim=0)
            p_cat = torch.cat([prior, prior], dim=0)

            eps_cat = base(x_cat, tB.repeat(2), c_cat, p_cat)
            eps_u, eps_c = eps_cat[:B], eps_cat[B:]
            eps = eps_u + cfg_scale * (eps_c - eps_u)

        # x0 stimato in spazio mm1
        x0_mm1 = (x - sqrt_1mab_t.to(x.dtype) * eps) / (sqrt_ab_t.to(x.dtype) + x.new_tensor(1e-8))
        x0_mm1 = x0_mm1.clamp(min=x.new_tensor(-MM1_CLAMP_EVAL), max=x.new_tensor(MM1_CLAMP_EVAL))

        # Proiezione: solo all'ultimo step (o ogni K)
        do_project = (t_prev is None) or (proj_every_k and ((j+1) % proj_every_k == 0))
        if do_project:
            x0_real = decode_target_torch(x0_mm1, y_log_scale)
            x0_proj = project_positive_with_sum(x0_real, psum_pred_real).clamp(min=0.0, max=y_clip_max)
            x0_mm1_used = (torch.log1p(x0_proj)/x0_proj.new_tensor(y_log_scale))*x0_proj.new_tensor(2.0) - x0_proj.new_tensor(1.0)
        else:
            x0_mm1_used = x0_mm1

        if t_prev is None:
            x = x0_mm1_used
            break

        # Passo precedente
        _, _, ab_prev = gather_ab(t_prev.expand(B), alpha_bars)
        sqrt_ab_prev   = torch.sqrt(ab_prev)
        sqrt_1mab_prev = torch.sqrt(1.0 - ab_prev)
        if eta == 0.0:
            x = sqrt_ab_prev.to(x.dtype) * x0_mm1_used + sqrt_1mab_prev.to(x.dtype) * eps
        else:
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
            sigma = sigma.view(B,1,1,1).to(x.dtype)
            noise = torch.randn_like(x)
            x = sqrt_ab_prev.to(x.dtype) * x0_mm1_used + torch.sqrt((sqrt_1mab_prev.to(x.dtype)**2 - sigma**2).clamp_min(0)) * eps + sigma * noise

        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

    return x  # [-1,1]

# -----------------------
# Metriche (CPU, batch)
# -----------------------
def entropy_normalized_batch(mats: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    B, H, W = mats.shape
    P = H * W
    flat = mats.reshape(B, P).astype(np.float64, copy=False)
    sums = flat.sum(axis=1, keepdims=True)
    out = np.zeros((B,), dtype=np.float32)
    mask = (sums[:,0] > 0)
    if np.any(mask):
        p = flat[mask] / np.maximum(sums[mask], eps)
        with np.errstate(divide='ignore'):
            plogp = p * np.log(p + eps)
        h = -plogp.sum(axis=1)
        out[mask] = (h / np.log(P)).astype(np.float32)
    return out

def gamma_max_batch(mats: np.ndarray) -> np.ndarray:
    return mats.reshape(mats.shape[0], -1).max(axis=1).astype(np.float32)

def gamma_tot_batch(mats: np.ndarray) -> np.ndarray:
    return mats.reshape(mats.shape[0], -1).sum(axis=1).astype(np.float32)

def auto_bins(x_real, x_fake, n_bins=60, qmax=99.5):
    xmax = np.nanmax([np.percentile(x_real, qmax), np.percentile(x_fake, qmax)])
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = max(np.max(x_real), np.max(x_fake))
    return np.linspace(0.0, float(xmax), n_bins+1)

# -----------------------
# Istogrammo 2D "spread" (ancorato a γ_max)
# -----------------------
def accumulate_spread_hist(batch: np.ndarray, hist_accum: np.ndarray, Xg: np.ndarray, Yg: np.ndarray):
    """
    batch: [B,H,W] float32 (>=0)
    hist_accum: [2H,2W] uint64 (contatori)
    Xg, Yg: griglie [H,W] con coordinate 0..W-1 / 0..H-1
    Regola: per pixel != 0 incrementa +1 nel bin (dx,dy) relativo al massimo
    """
    B, H, W = batch.shape
    H2, W2 = 2*H, 2*W
    flat = batch.reshape(B, -1)
    imax = flat.argmax(axis=1)                # [B]
    y_max = (imax // W).astype(np.int32)
    x_max = (imax %  W).astype(np.int32)

    for b in range(B):
        m = batch[b] != 0.0
        if not np.any(m): continue
        dx = (Xg - x_max[b])[m]
        dy = (Yg - y_max[b])[m]
        ix = (dx + W).astype(np.int32)
        iy = (dy + H).astype(np.int32)
        if ix.size:
            np.clip(ix, 0, W2-1, out=ix)
            np.clip(iy, 0, H2-1, out=iy)
            np.add.at(hist_accum, (iy, ix), 1)

# -----------------------
# MAIN
# -----------------------
def main():
    # --- CKPT & cfg ---
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg  = ckpt.get("cfg", {})
    IMG_SIZE       = int(cfg.get("IMG_SIZE", 44))
    T_STEPS        = int(cfg.get("T_STEPS", 1000))
    TIME_DIM       = int(cfg.get("TIME_DIM", 64))
    COND_EMB       = int(cfg.get("COND_EMB", 64))
    BASE_CH        = int(cfg.get("BASE_CH", 64))
    PRIOR_SIGMA_PX = float(cfg.get("PRIOR_SIGMA_PX", 2.5))
    COND7_DIM      = 7
    COND_DIM       = COND7_DIM + 3

    cond7_mean = ckpt["cond7_mean"].float().to(DEVICE)
    cond7_std  = ckpt["cond7_std"].float().to(DEVICE)
    psum_mean  = float(ckpt["psum_mean"])
    psum_std   = float(ckpt["psum_std"])
    y_clip_max = float(ckpt["y_clip_max"])
    y_log_scale= float(ckpt["y_log_scale"])
    _, _, alpha_bars = cosine_alpha_bars(T_STEPS, device=DEVICE, dtype=torch.float32)

    # --- istanzia ---
    unet = CondUNet(cond_dim=COND_DIM, time_dim=TIME_DIM, cond_emb=COND_EMB, base_ch=BASE_CH).to(DEVICE)
    pred = GlobalPredictor(in_dim=COND7_DIM, hid=128).to(DEVICE)
    unet = unet.to(memory_format=torch.channels_last)

    # --- carica pesi EMA dal ckpt (non compilato) ---
    unet.load_state_dict(ckpt["unet_ema"], strict=True)
    pred.load_state_dict(ckpt["pred_ema"], strict=True)

    # --- ora compila (facoltativo) ---
    try:
        unet = torch.compile(unet, mode="reduce-overhead", fullgraph=False)
        pred = torch.compile(pred, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

    # --- eval dopo la (eventuale) compilazione ---
    unet.eval(); pred.eval()


    # --- Test set ---
    df = load_test_df(DATA_DIR)
    cond7_cols, psum_col, image_col = infer_columns(df)
    cond7_np  = df[cond7_cols].to_numpy(dtype=np.float32, copy=False)  # [N,7]
    img_list  = df[image_col].tolist()                                 # lista di np.array 44x44
    N = len(img_list)
    H = W = IMG_SIZE
    print(f"[DATA] Test samples: {N}")

    # Griglie coordinate 0..W-1 / 0..H-1 (per offsets veloci)
    Xg = np.tile(np.arange(W, dtype=np.int32)[None, :], (H, 1))
    Yg = np.tile(np.arange(H, dtype=np.int32)[:, None], (1, W))

    # --- Precompute cond7_n (GPU) ---
    cond7_t_all = torch.from_numpy(cond7_np).to(DEVICE)
    cond7_t_all = (cond7_t_all - cond7_mean[None,:]) / (cond7_std[None,:] + 1e-6)

    # --- Vettori metriche 1D ---
    gmax_real = np.empty((N,), dtype=np.float32)
    gmax_fake = np.empty((N,), dtype=np.float32)
    gtot_real = np.empty((N,), dtype=np.float32)
    gtot_fake = np.empty((N,), dtype=np.float32)
    hent_real = np.empty((N,), dtype=np.float32)
    hent_fake = np.empty((N,), dtype=np.float32)

    # --- Istogrammi 2D spread ---
    hist_real = np.zeros((2*H, 2*W), dtype=np.uint64)
    hist_fake = np.zeros((2*H, 2*W), dtype=np.uint64)

    saved = 0

    # --- Loop batch ---
    for start in range(0, N, BATCH_GEN):
        end = min(N, start + BATCH_GEN)
        B = end - start

        # [TIMING] batch timer totale
        t_batch0 = time.perf_counter()

        # real batch (CPU) — stack per batch e clip
        real_batch = np.stack([np.asarray(img_list[i], dtype=np.float32) for i in range(start, end)], axis=0)
        np.clip(real_batch, 0.0, y_clip_max, out=real_batch)

        # cond7 batch (GPU)
        cond7 = cond7_t_all[start:end]

        with torch.inference_mode():
            # Stadio A: predictor (NON incluso in t_gen di default)
            ps_pred_n, xy_pred_u = pred(cond7)
            psum_pred_real = torch.expm1(ps_pred_n.squeeze(1) * psum_std + psum_mean)
            psum_pred_real = torch.nan_to_num(psum_pred_real, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=0.0)
            cond_vec = torch.cat([cond7, ps_pred_n, xy_pred_u], dim=1).to(torch.float32)

            # [TIMING] sincronizza e misura SOLO sampling+decode
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_gen0 = time.perf_counter()

            x_mm1 = ddim_sample_cfg_fast(
                unet, cond_vec,
                xy_pred_u[:,0], xy_pred_u[:,1], psum_pred_real,
                alpha_bars, T_STEPS, IMG_SIZE, PRIOR_SIGMA_PX,
                y_log_scale, y_clip_max,
                steps=EVAL_STEPS, cfg_scale=CFG_SCALE, eta=ETA, proj_every_k=PROJ_EVERY_K
            )
            gen_real_t = decode_target_torch(x_mm1, y_log_scale).clamp(min=0.0, max=y_clip_max)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_gen1 = time.perf_counter()

        # CPU arrays (rilascio VRAM subito dopo)
        gen_batch = gen_real_t.squeeze(1).detach().cpu().numpy()
        del x_mm1, gen_real_t, cond7, ps_pred_n, xy_pred_u, psum_pred_real, cond_vec
        torch.cuda.empty_cache(); gc.collect()

        # [TIMING] inizio fase di ELABORAZIONE (metriche + istogrammi)
        t_proc0 = time.perf_counter()

        # Metriche per batch
        gmax_real[start:end] = gamma_max_batch(real_batch)
        gtot_real[start:end] = gamma_tot_batch(real_batch)
        hent_real[start:end] = entropy_normalized_batch(real_batch)

        gmax_fake[start:end] = gamma_max_batch(gen_batch)
        gtot_fake[start:end] = gamma_tot_batch(gen_batch)
        hent_fake[start:end] = entropy_normalized_batch(gen_batch)

        # Istogrammi 2D spread (solo pixel != 0)
        accumulate_spread_hist(real_batch, hist_real, Xg, Yg)
        accumulate_spread_hist(gen_batch,  hist_fake, Xg, Yg)

        # opzionale: salva qualche coppia
        if SAVE_SAMPLE_K > 0 and saved < SAVE_SAMPLE_K:
            for i in range(min(SAVE_SAMPLE_K - saved, B)):
                np.save(f"sample_real_{start+i:06d}.npy", real_batch[i])
                np.save(f"sample_fake_{start+i:06d}.npy", gen_batch[i])
                saved += 1

        # libera batch
        del real_batch, gen_batch
        gc.collect()

        # [TIMING] fine elaborazione e batch
        t_proc1  = time.perf_counter()
        t_batch1 = time.perf_counter()

        t_gen   = t_gen1  - t_gen0                     # sampling+decode
        t_proc  = t_proc1 - t_proc0                    # metriche/ist
        t_batch = t_batch1 - t_batch0                  # totale batch

        ms_per_item_gen  = (t_gen  / B) * 1000.0
        ms_per_item_proc = (t_proc / B) * 1000.0
        ms_per_item_tot  = (t_batch/ B) * 1000.0

        print(
            f"[GEN] {end}/{N} | B={B} | "
            f"t_gen={t_gen:.2f}s ({ms_per_item_gen:.1f} ms/itm) | "
            f"t_proc={t_proc:.2f}s ({ms_per_item_proc:.1f} ms/itm) | "
            f"t_tot={t_batch:.2f}s ({ms_per_item_tot:.1f} ms/itm)"
        )

    # --- CSV finale ---
    out_csv = "ddpm_test_distributions.csv"
    pd.DataFrame({
        "idx": np.arange(N, dtype=int),
        "gamma_max_real": gmax_real,
        "gamma_max_fake": gmax_fake,
        "gamma_tot_real": gtot_real,
        "gamma_tot_fake": gtot_fake,
        "entropy_norm_real": hent_real,
        "entropy_norm_fake": hent_fake,
    }).to_csv(out_csv, index=False)
    print(f"[OK] Salvato CSV: {out_csv}")

    # --- Plot istogrammi 1D ---
    def plot_and_save(xr, xf, title, xlabel, fname, bins=None):
        if bins is None:
            bins = auto_bins(xr, xf, n_bins=60, qmax=99.5)
        plt.figure(figsize=(7,4))
        plt.hist(xr, bins=bins, alpha=0.55, label="Real", edgecolor="none")
        plt.hist(xf, bins=bins, alpha=0.55, label="Generated", edgecolor="none")
        plt.xlabel(xlabel); plt.ylabel("Frequenza"); plt.title(title)
        plt.legend(); plt.tight_layout(); plt.savefig(fname, dpi=140); plt.close()
        print(f"[OK] Salvato: {fname}")

    plot_and_save(gmax_real, gmax_fake, "Distribuzione di γ_max — Real vs Generated",
                  "γ_max (valore massimo per matrice)", "ddpm_dist_gamma_max.png")
    plot_and_save(gtot_real, gtot_fake, "Distribuzione di γ_tot — Real vs Generated",
                  "γ_tot (somma dei pixel)", "ddpm_dist_gamma_tot.png")

    bins_h = np.linspace(0.0, 1.0, 51)
    plot_and_save(hent_real, hent_fake, "Distribuzione entropia — Real vs Generated",
                  "Entropia normalizzata (0=concentrata, 1=uniforme)", "ddpm_dist_entropy.png", bins=bins_h)

    # --- Plot istogrammi 2D spread (affiancati) ---
    from matplotlib.colors import LogNorm
    extent = [-W, W, -H, H]  # (0,0) al centro
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    if USE_LOG_IMSHOW:
        im0 = axes[0].imshow(hist_real.astype(np.float64), origin="lower", extent=extent,
                             norm=LogNorm(vmin=1, vmax=float(hist_real.max() or 1)))
        im1 = axes[1].imshow(hist_fake.astype(np.float64), origin="lower", extent=extent,
                             norm=LogNorm(vmin=1, vmax=float(hist_fake.max() or 1)))
    else:
        im0 = axes[0].imshow(hist_real, origin="lower", extent=extent)
        im1 = axes[1].imshow(hist_fake, origin="lower", extent=extent)
    axes[0].set_title("Spread 2D — REAL"); axes[1].set_title("Spread 2D — GENERATED")
    for ax in axes:
        ax.set_xlabel("Δx (pixel)"); ax.set_ylabel("Δy (pixel)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.5); ax.axvline(0, color="k", lw=0.5, alpha=0.5)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="conteggi")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="conteggi")
    plt.savefig("ddpm_hist2d_spread_real_vs_generated.png", dpi=150)
    plt.close(fig)
    print("[OK] Salvato: ddpm_hist2d_spread_real_vs_generated.png")

    # --- Proiezioni 1D (x e y) ---
    x_coords = np.arange(-W, W, dtype=np.int32)
    y_coords = np.arange(-H, H, dtype=np.int32)
    projx_real = hist_real.sum(axis=0)
    projx_fake = hist_fake.sum(axis=0)
    projy_real = hist_real.sum(axis=1)
    projy_fake = hist_fake.sum(axis=1)

    plt.figure(figsize=(7.5,4))
    plt.plot(x_coords, projx_real, label="Real")
    plt.plot(x_coords, projx_fake, label="Generated")
    plt.xlabel("Δx (pixel)"); plt.ylabel("Conteggi"); plt.title("Proiezione su Δx — Spread 2D")
    plt.legend(); plt.tight_layout(); plt.savefig("ddpm_hist2d_proj_x.png", dpi=150); plt.close()
    print("[OK] Salvato: ddpm_hist2d_proj_x.png")

    plt.figure(figsize=(7.5,4))
    plt.plot(y_coords, projy_real, label="Real")
    plt.plot(y_coords, projy_fake, label="Generated")
    plt.xlabel("Δy (pixel)"); plt.ylabel("Conteggi"); plt.title("Proiezione su Δy — Spread 2D")
    plt.legend(); plt.tight_layout(); plt.savefig("ddpm_hist2d_proj_y.png", dpi=150); plt.close()
    print("[OK] Salvato: ddpm_hist2d_proj_y.png")

if __name__ == "__main__":
    main()

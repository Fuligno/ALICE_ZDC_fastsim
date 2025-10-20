# ============================================================
# DDPM_infer_compare.py — Inference + confronto 4x2 (real vs generated)
# - Carica best_ema_checkpoint.pt (o altro ckpt compatibile)
# - Legge test.pkl/.plk, standardizza cond7 come in training (da ckpt)
# - Predice psum/XY (stadio A), campiona con DDIM+CFG (stadio B)
# - Plot: 4 righe (scelte a caso), 2 colonne (reale, generata), colorbar per ciascun ass, z-scale indipendente
# ============================================================
import os, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------
# CONFIG (modifica qui)
# -----------------------
DATA_DIR   = "/data/dataalice/dfuligno/tesi/myTesi/Dati_processed"  # cartella che contiene test.pkl
CKPT_PATH  = "DDPM_stage_checkpoints_V1/best_ema_checkpoint.pt"                                # pesi EMA salvati durante il training
NUM_ROWS   = 4
EVAL_STEPS = 50
CFG_SCALE  = 3.0
ETA        = 0.0
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.float16
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.benchmark = True
try: torch.set_float32_matmul_precision("high")
except: pass

# -----------------------
# UTILS DATA
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
        # psum probabili
        for cand in ["photonSum", "psum", "PhotonSum", "psum_total"]:
            if cand in cols:
                psum_col = cand
                break
        else:
            psum_col = cols[8]  # fallback
        cond7_cols = cols[1:8]
    else:
        # schema base: cond7=cols[1:8], psum=cols[8], image=cols[9] (colonna 10)
        cond7_cols = cols[1:8]
        psum_col   = cols[8]
        image_col  = cols[9]
    return cond7_cols, psum_col, image_col

def argmax_xy_unit_np(batch_np: np.ndarray):
    N, H, W = batch_np.shape
    flat_idx = batch_np.reshape(N,-1).argmax(axis=1)
    ys = (flat_idx // W).astype(np.float32)
    xs = (flat_idx %  W).astype(np.float32)
    return (xs / np.float32(W - 1)).astype(np.float32), (ys / np.float32(H - 1)).astype(np.float32)

# -----------------------
# DIFFUSION SCHEDULE
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
# MODEL COMPONENTS
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
# ENCODE/DECODE & PHYS HELPERS
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
# DDIM SAMPLER (CFG + proiezione somma)
# -----------------------
@torch.no_grad()
def ddim_sample_cfg(unet_model, cond_vec, x_u, y_u, psum_pred_real, alpha_bars, T_STEPS, IMG_SIZE, PRIOR_SIGMA_PX, steps=150, cfg_scale=3.0, eta=0.0):
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
            eps_u = base(x, t.expand(B), zeros_cond, prior)
            eps_c = base(x, t.expand(B), cond_vec,   prior)
            eps   = eps_u + cfg_scale * (eps_c - eps_u)

        x0_mm1 = (x - sqrt_1mab_t.to(x.dtype) * eps) / (sqrt_ab_t.to(x.dtype) + x.new_tensor(1e-8))
        x0_mm1 = x0_mm1.clamp(min=x.new_tensor(-MM1_CLAMP_EVAL), max=x.new_tensor(MM1_CLAMP_EVAL))

        # decoding -> proiezione -> re-encode (per step successivo)
        # (la re-encode qui è nello spazio mm1 coerente con il training)
        # verrà ridotta all'ultimo step
        # NOTA: la proiezione su somma reale avviene su x0 nello spazio reale
        # (la somma prevista è psum_pred_real)
        # (il clipping a y_clip_max lo applichiamo dopo la decodifica finale nel chiamante)
        # -> qui rimaniamo nello spazio di x0_mm1
        # In pratica: usiamo x0_proj solo per calcolare x al passo precedente
        # e non imponiamo altri vincoli.
        # Per coerenza con il training, basta questo.
        if t_prev is None:
            x = x0_mm1
            break

        _, _, ab_prev = gather_ab(t_prev.expand(B), alpha_bars)
        sqrt_ab_prev   = torch.sqrt(ab_prev)
        sqrt_1mab_prev = torch.sqrt(1.0 - ab_prev)
        if eta == 0.0:
            x = sqrt_ab_prev.to(x.dtype) * x0_mm1 + sqrt_1mab_prev.to(x.dtype) * eps
        else:
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
            sigma = sigma.view(B,1,1,1).to(x.dtype)
            noise = torch.randn_like(x)
            x = sqrt_ab_prev.to(x.dtype) * x0_mm1 + torch.sqrt((sqrt_1mab_prev.to(x.dtype)**2 - sigma**2).clamp_min(0)) * eps + sigma * noise

        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

    return x  # [-1,1]

# -----------------------
# MAIN
# -----------------------
def main():
    # --- Carica checkpoint ---
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg  = ckpt.get("cfg", {})
    IMG_SIZE       = int(cfg.get("IMG_SIZE", 44))
    T_STEPS        = int(cfg.get("T_STEPS", 1000))
    TIME_DIM       = int(cfg.get("TIME_DIM", 64))
    COND_EMB       = int(cfg.get("COND_EMB", 64))
    BASE_CH        = int(cfg.get("BASE_CH", 64))
    PRIOR_SIGMA_PX = float(cfg.get("PRIOR_SIGMA_PX", 2.5))
    COND7_DIM      = 7
    COND_DIM       = COND7_DIM + 3  # (psum_n, x_u, y_u)

    # stats & scalings dal ckpt (salvati nel training loop)
    cond7_mean = ckpt["cond7_mean"].float().to(DEVICE)   # [7]
    cond7_std  = ckpt["cond7_std"].float().to(DEVICE)    # [7]
    psum_mean  = float(ckpt["psum_mean"])
    psum_std   = float(ckpt["psum_std"])
    y_clip_max = float(ckpt["y_clip_max"])
    y_log_scale= float(ckpt["y_log_scale"])

    # schedule
    _, _, alpha_bars = cosine_alpha_bars(T_STEPS, device=DEVICE, dtype=torch.float32)

    # --- Istanzia modelli e carica pesi EMA ---
    unet = CondUNet(cond_dim=COND_DIM, time_dim=TIME_DIM, cond_emb=COND_EMB, base_ch=BASE_CH).to(DEVICE)
    pred = GlobalPredictor(in_dim=COND7_DIM, hid=128).to(DEVICE)
    unet = unet.to(memory_format=torch.channels_last)
    unet.load_state_dict(ckpt["unet_ema"], strict=True)
    pred.load_state_dict(ckpt["pred_ema"], strict=True)
    unet.eval(); pred.eval()

    # --- Carica test set e normalizza cond7/psum come in training ---
    df = load_test_df(DATA_DIR)
    cond7_cols, psum_col, image_col = infer_columns(df)
    cond7_np = df[cond7_cols].to_numpy(dtype=np.float32, copy=False)
    psum_np  = df[psum_col].to_numpy(dtype=np.float32, copy=False)
    mats_np  = np.stack([np.asarray(x, dtype=np.uint16) for x in df[image_col].tolist()], axis=0)  # [N,44,44]
    N = mats_np.shape[0]
    assert N >= NUM_ROWS, f"Il test set ha {N} righe, servono almeno {NUM_ROWS}."

    # standardizzazione cond7 / psum_log
    cond7_t = torch.from_numpy(cond7_np).to(DEVICE)
    cond7_t = (cond7_t - cond7_mean[None,:]) / (cond7_std[None,:] + 1e-6)
    psum_log_t = torch.log1p(torch.from_numpy(psum_np).to(DEVICE))
    psum_n_t   = (psum_log_t - psum_mean) / (psum_std + 1e-6)

    # --- Estrai 4 indici casuali ---
    idxs = np.random.choice(N, size=NUM_ROWS, replace=False)
    idxs_t = torch.from_numpy(idxs).long().to(DEVICE)

    # batch selezionato
    cond7_sel = cond7_t[idxs_t]                # [B,7]
    # Stadio A: predizione psum_n e XY (unit)
    with torch.no_grad():
        ps_pred_n, xy_pred_u = pred(cond7_sel)     # [B,1], [B,2]
        psum_pred_real = torch.expm1(ps_pred_n.squeeze(1) * psum_std + psum_mean)
        psum_pred_real = torch.nan_to_num(psum_pred_real, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=0.0)

        cond_vec = torch.cat([cond7_sel, ps_pred_n, xy_pred_u], dim=1).to(torch.float32)

        # Stadio B: campionamento DDIM+CFG
        x_mm1 = ddim_sample_cfg(
            unet, cond_vec,
            xy_pred_u[:,0], xy_pred_u[:,1], psum_pred_real,
            alpha_bars, T_STEPS, IMG_SIZE, PRIOR_SIGMA_PX,
            steps=EVAL_STEPS, cfg_scale=CFG_SCALE, eta=ETA
        )
        gen_real = decode_target_torch(x_mm1, y_log_scale).clamp(min=0.0, max=y_clip_max).squeeze(1).detach().cpu().numpy()  # [B,44,44]

    # reali corrispondenti
    real_sel = mats_np[idxs]  # uint16
    real_sel = np.clip(real_sel.astype(np.float32), 0.0, y_clip_max)

    # --- Plot 4x2 con colorbar per ogni ass (z-scale indipendente) ---
    fig, axes = plt.subplots(NUM_ROWS, 2, figsize=(8, 2.4*NUM_ROWS), constrained_layout=True)
    if NUM_ROWS == 1:
        axes = np.array([axes])  # shape (1,2)

    for r in range(NUM_ROWS):
        # reale
        ax = axes[r, 0]
        im = ax.imshow(real_sel[r], origin="lower", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Real (idx={idxs[r]})")
        ax.set_xticks([]); ax.set_yticks([])

        # generata
        ax = axes[r, 1]
        im = ax.imshow(gen_real[r], origin="lower", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Generated (idx={idxs[r]})")
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("DDPM — Test comparison", y=1.02, fontsize=12)
    out_path = "ddpm_inference_comparison.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[OK] Salvato: {out_path}")

if __name__ == "__main__":
    main()

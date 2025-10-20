# utility.py
# Utilities comuni per preprocessing e training (dataset ZDC shardati .pkl)

from __future__ import annotations
import os, math, json, gzip
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Iterable, List

import numpy as np
import pandas as pd

# =============== RNG ===============
def seed_everything(seed: int = 123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

# =============== Scaler semplice (numpy) ===============
@dataclass
class StandardScalerNP:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    eps: float = 1e-8
    def fit(self, X: np.ndarray) -> "StandardScalerNP":
        X = np.asarray(X)
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0) + self.eps
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X) - self.mean) / self.std
    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z) * self.std + self.mean
    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "eps": self.eps}
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StandardScalerNP":
        return StandardScalerNP(mean=np.array(d["mean"]), std=np.array(d["std"]), eps=d.get("eps", 1e-8))

# =============== Feature fisiche ===============
def momentum_to_angles(px: float, py: float, pz: float) -> Tuple[float, float, float]:
    p = math.sqrt(px*px + py*py + pz*pz) + 1e-12
    theta = math.acos(max(-1.0, min(1.0, pz/p)))
    phi = math.atan2(py, px)
    return theta, phi, p

def unit_vector(px: float, py: float, pz: float) -> Tuple[float, float, float, float]:
    p = math.sqrt(px*px + py*py + pz*pz) + 1e-12
    return px/p, py/p, pz/p, p

def compute_impact(image: np.ndarray, method: str = "centroid") -> Tuple[float, float, float, float]:
    """
    Ritorna (x_u, y_u, peak, total) — x_u,y_u normalizzati in [0,1].
    method: 'centroid' (default) o 'argmax'
    """
    assert image.ndim == 2, "image deve essere 2D [H,W]"
    H, W = image.shape
    total = float(image.sum())
    peak = float(image.max())
    if method == "argmax" or total <= 0:
        idx = int(image.argmax())
        y = idx // W; x = idx % W
        return x/(W-1+1e-12), y/(H-1+1e-12), peak, total
    ys, xs = np.indices((H, W))
    cx = float((xs*image).sum()/(total+1e-12))
    cy = float((ys*image).sum()/(total+1e-12))
    return cx/(W-1+1e-12), cy/(H-1+1e-12), peak, total

def spot_sigma_from_angles(theta: float, phi: float, base_sigma_px: float = 2.0) -> float:
    # euristica: più inclinato ⇒ spot più largo
    k = 0.8
    return float(base_sigma_px * (1.0 + k * abs(math.sin(theta))))

# =============== I/O generico ===============
def save_json_gz(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

# =============== LMDB immagini (opzionale) ===============
def write_images_to_lmdb(lmdb_path: str, images: Iterable[np.ndarray], map_size_gb: float = 2.0):
    """
    Salva sequenza di immagini 2D float32 in LMDB (chiavi 00000000, 00000001, ...).
    """
    import lmdb
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=int(map_size_gb*(1024**3)))
    with env.begin(write=True) as txn:
        for idx, img in enumerate(images):
            key = f"{idx:08d}".encode("ascii")
            txn.put(key, img.astype(np.float32).tobytes())
        txn.put(b"__meta__", json.dumps({"dtype":"float32"}).encode("utf-8"))
    env.sync(); env.close()

def read_image_from_lmdb(lmdb_path: str, idx: int, H: int, W: int) -> np.ndarray:
    import lmdb, numpy as np
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        buf = txn.get(f"{idx:08d}".encode("ascii"))
        if buf is None: raise KeyError(f"Idx {idx} non trovato in {lmdb_path}")
        arr = np.frombuffer(buf, dtype=np.float32).reshape(H, W)
    env.close()
    return arr

# =============== Supporto a .pkl/.plk (DataFrame) ===============
def infer_columns_pkl(df: pd.DataFrame) -> Tuple[List[str], str, str]:
    """
    Ritorna (cond7_cols, psum_col, image_col) inferiti dal DF.
    Regole compatibili con il tuo codice storico.
    """
    cols = list(df.columns)
    # preferisci colonna 'image' se presente
    if "image" in cols:
        image_col = "image"
    else:
        # fallback: 10a colonna (indice 9) nel tuo schema classico
        image_col = cols[9]

    # psum nomi tipici
    for cand in ("photonSum","psum","PhotonSum","psum_total"):
        if cand in cols:
            psum_col = cand; break
    else:
        psum_col = cols[8]  # fallback

    # cond7: spesso sono le 7 colonne in posizione 1..7
    if len(cols) >= 8:
        cond7_cols = cols[1:8]
    else:
        # altrimenti prova a trovare 7 float columns diverse da image/psum
        cand = [c for c in cols if c not in (image_col, psum_col)]
        cond7_cols = cand[:7]
    return cond7_cols, psum_col, image_col


def incidence_angle_from_momentum(px: float, py: float, pz: float,
                                  nx: float = 0.0, ny: float = 0.0, nz: float = 1.0) -> float:
    """
    Angolo di incidenza rispetto alla normale del rivelatore n=(nx,ny,nz).
    Ritorna theta_inc in [0, pi].
    """
    # versore del momento
    p = math.sqrt(px*px + py*py + pz*pz) + 1e-12
    ux, uy, uz = px/p, py/p, pz/p
    # normalizza n
    nn = math.sqrt(nx*nx + ny*ny + nz*nz) + 1e-12
    nx, ny, nz = nx/nn, ny/nn, nz/nn
    # cos(theta_inc) = u · n
    c = max(-1.0, min(1.0, ux*nx + uy*ny + uz*nz))
    return math.acos(c)
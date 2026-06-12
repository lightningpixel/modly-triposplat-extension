"""
SplatForge Phase A geometry optimizer — ALPHA variant (fidelity-first).

Moves mesh vertices onto the iso-surface of a GaussianField using Adam +
cosine LR decay. Regularisation is minimal: tiny edge-variance and Laplacian
weights ensure the geometry stays valid while fidelity terms dominate.

Public API
----------
DEFAULT_WEIGHTS : dict
optimize_phase_a(verts, faces, field, n_steps, lr, weights, device, budget_s)
    -> (verts_np float32, faces_np int64)
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_EXT = Path(__file__).parent.parent.parent  # extension root
if str(_EXT) not in sys.path:
    sys.path.insert(0, str(_EXT))

from splatforge.fields import GaussianField  # noqa: E402
from splatforge.remesh import remesh_step    # noqa: E402


# --------------------------------------------------------------------------- #
# Default weight presets (ALPHA — fidelity-first)
# --------------------------------------------------------------------------- #

DEFAULT_WEIGHTS: dict = {
    "Refined": {
        "w_iso":       1.0,
        "w_normal":    0.3,
        "w_edge":      0.02,
        "w_lap":       0.005,
        "remesh_every": 40,
    },
    "Maximum": {
        "w_iso":       1.0,
        "w_normal":    0.3,
        "w_edge":      0.02,
        "w_lap":       0.005,
        "remesh_every": 40,
    },
}


# --------------------------------------------------------------------------- #
# Differentiable geometry helpers (O(V) memory — no (V,V) matrices)
# --------------------------------------------------------------------------- #

def _vertex_normals(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals, differentiable w.r.t. v.

    v: (V,3)  f: (F,3) int64
    Returns: (V,3) unit normals.
    """
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = torch.linalg.cross(v1 - v0, v2 - v0)  # (F,3) face normals (area-weighted)
    vn = torch.zeros_like(v)
    vn.scatter_add_(0, f[:, 0:1].expand(-1, 3), fn)
    vn.scatter_add_(0, f[:, 1:2].expand(-1, 3), fn)
    vn.scatter_add_(0, f[:, 2:3].expand(-1, 3), fn)
    return F.normalize(vn, dim=1)


def _laplacian_loss(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Mean-squared umbrella-Laplacian displacement, differentiable w.r.t. v.

    Uses scatter_add_ — O(E) memory, no (V,V) matrix.
    """
    nv = v.shape[0]
    # Each undirected edge (i,j) contributes two directed pairs
    src = torch.cat([f[:, 0], f[:, 1], f[:, 2],
                     f[:, 1], f[:, 2], f[:, 0]])
    dst = torch.cat([f[:, 1], f[:, 2], f[:, 0],
                     f[:, 0], f[:, 1], f[:, 2]])
    nbr_sum = torch.zeros_like(v)
    nbr_cnt = torch.zeros(nv, 1, device=v.device, dtype=v.dtype)
    nbr_sum.scatter_add_(0, src.unsqueeze(1).expand(-1, 3), v[dst])
    nbr_cnt.scatter_add_(0, src.unsqueeze(1),
                         torch.ones(len(src), 1, device=v.device, dtype=v.dtype))
    lap = v - nbr_sum / nbr_cnt.clamp_min(1.0)
    return (lap ** 2).sum(1).mean()


def _edge_var_loss(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Variance of per-face edge lengths (cheap proxy).

    Differentiable w.r.t. v.  Avoids building a full edge list.
    """
    elen = (v[f[:, 1]] - v[f[:, 0]]).norm(dim=1)
    return elen.var()


# --------------------------------------------------------------------------- #
# Mean edge length helper (numpy, for step clamping)
# --------------------------------------------------------------------------- #

def _mean_edge_length_np(verts_np: np.ndarray, faces_np: np.ndarray) -> float:
    """Average length of face edges (all three per face, cheap)."""
    v0 = verts_np[faces_np[:, 0]]
    v1 = verts_np[faces_np[:, 1]]
    e01 = np.linalg.norm(v1 - v0, axis=1)
    return float(e01.mean()) if len(e01) > 0 else 1.0


# --------------------------------------------------------------------------- #
# Main optimiser
# --------------------------------------------------------------------------- #

def optimize_phase_a(
    verts: np.ndarray,
    faces: np.ndarray,
    field: "GaussianField",
    n_steps: int = 200,
    lr: float = 2e-3,
    weights: dict = None,
    device: torch.device = None,
    budget_s: float = 60.0,
) -> tuple:
    """Phase A geometry optimiser — ALPHA (fidelity-first) variant.

    Parameters
    ----------
    verts      : (V,3) float32 numpy array — mesh vertices (splat/Z-up frame)
    faces      : (F,3) int64  numpy array — triangle faces
    field      : GaussianField — provides density(), density_grad(), tau, device
    n_steps    : total Adam iterations
    lr         : initial learning rate
    weights    : dict with keys w_iso, w_normal, w_edge, w_lap, remesh_every.
                 Defaults to DEFAULT_WEIGHTS["Refined"] if None.
    device     : torch.device; falls back to field.device if None
    budget_s   : hard wall-clock budget in seconds

    Returns
    -------
    (verts_np float32, faces_np int64)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS["Refined"]

    w_iso        = float(weights.get("w_iso",        1.0))
    w_normal     = float(weights.get("w_normal",     0.3))
    w_edge       = float(weights.get("w_edge",       0.02))
    w_lap        = float(weights.get("w_lap",        0.005))
    remesh_every = int(weights.get("remesh_every",   40))

    if device is None:
        device = field.device

    tau = field.tau

    # ------------------------------------------------------------------ #
    # Helpers: build torch tensors from numpy arrays
    # ------------------------------------------------------------------ #

    def _to_tensors(v_np: np.ndarray, f_np: np.ndarray):
        v = torch.tensor(v_np, dtype=torch.float32, device=device,
                         requires_grad=True)
        f = torch.tensor(f_np, dtype=torch.long, device=device)
        return v, f

    # ------------------------------------------------------------------ #
    # Initialise
    # ------------------------------------------------------------------ #

    verts_np = np.array(verts, dtype=np.float32)
    faces_np = np.array(faces, dtype=np.int64)

    v, f = _to_tensors(verts_np, faces_np)

    opt = torch.optim.Adam([v], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps,
                                                            eta_min=lr * 0.01)

    best_loss  = float("inf")
    best_verts = verts_np.copy()
    best_faces = faces_np.copy()

    t0 = time.time()

    for step in range(n_steps):
        # Budget guard
        if time.time() - t0 > budget_s:
            break

        # ---------------------------------------------------------------- #
        # Remesh (topology refresh)
        # ---------------------------------------------------------------- #
        if remesh_every > 0 and step > 0 and step % remesh_every == 0:
            v_np_cur = v.detach().cpu().numpy().astype(np.float32)
            f_np_cur = f.cpu().numpy().astype(np.int64)
            try:
                v_np_new, f_np_new = remesh_step(v_np_cur, f_np_cur, field)
                v_np_new = v_np_new.astype(np.float32)
                f_np_new = f_np_new.astype(np.int64)
                if len(v_np_new) > 0 and len(f_np_new) > 0:
                    v, f = _to_tensors(v_np_new, f_np_new)
                    field.refresh_knn_cache()  # vertex set changed → stale KNN
                    remaining = n_steps - step
                    opt = torch.optim.Adam([v], lr=scheduler.get_last_lr()[0])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=max(remaining, 1), eta_min=lr * 0.01)
            except Exception:
                pass  # remesh failure is non-fatal; keep current mesh

        # ---------------------------------------------------------------- #
        # Forward pass
        # ---------------------------------------------------------------- #
        opt.zero_grad()

        rho = field.density(v)                             # (V,) differentiable
        L_iso = ((rho - tau) ** 2).mean()

        vn = _vertex_normals(v, f)                         # (V,3) differentiable
        g  = field.density_grad(v)                         # (V,3) detached (analytical)
        g_hat = F.normalize(g, dim=1)
        L_normal = (1.0 - (vn * g_hat).sum(1).abs()).mean()

        L_edge = _edge_var_loss(v, f)
        L_lap  = _laplacian_loss(v, f)

        loss = (w_iso    * L_iso
                + w_normal * L_normal
                + w_edge   * L_edge
                + w_lap    * L_lap)

        # NaN guard
        if loss.isnan():
            break

        # Snapshot positions before the optimizer mutates them
        with torch.no_grad():
            v_prev = v.detach().clone()
            f_np_snap = f.cpu().numpy().astype(np.int64)
            v_np_snap = v_prev.cpu().numpy().astype(np.float32)
            mel = _mean_edge_length_np(v_np_snap, f_np_snap)
            max_step = 0.5 * mel

        loss.backward()
        opt.step()
        scheduler.step()

        # ---------------------------------------------------------------- #
        # Step clamping: limit per-step displacement to 0.5 × mean_edge_len
        # ---------------------------------------------------------------- #
        with torch.no_grad():
            delta     = v.data - v_prev                            # (V,3)
            dnorm     = delta.norm(dim=1, keepdim=True)            # (V,1)
            scale     = (max_step / dnorm.clamp_min(1e-8)).clamp_max(1.0)
            v.data.copy_(v_prev + delta * scale)

        # ---------------------------------------------------------------- #
        # Track best
        # ---------------------------------------------------------------- #
        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss  = loss_val
            best_verts = v.detach().cpu().numpy().astype(np.float32)
            best_faces = f.cpu().numpy().astype(np.int64)

    return best_verts, best_faces

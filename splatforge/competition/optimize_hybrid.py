"""
SplatForge Phase A — HYBRID optimizer (ALPHA fidelity + cotangent Laplacian).

ALPHA base weights (w_iso=1.0, w_normal=0.3, w_edge=0.02) with a geometry-aware
cotangent Laplacian replacing the uniform umbrella. Key invariants:

  - Cotan weights clamped to ≥0: avoids obtuse-angle sign-flip instability.
  - Per-vertex fallback to uniform weights when all cotan weights sum to zero
    (fully-obtuse diamond patch).
  - Weights recomputed only after each remesh pass (cached between steps).
    The topology (src/dst indices + cotan values) is fixed between remesh passes;
    only vertex positions are optimized. This saves O(F) cotan computation per step.
  - remesh_every=40 (ALPHA cadence, not BETA's 20) to stay within 90-s budget.

Per-category profiling printed at end if profile=True.

Public API
----------
DEFAULT_WEIGHTS : dict
optimize_phase_a(verts, faces, field, n_steps, lr, weights, device, budget_s,
                 profile) -> (verts_np float32, faces_np int64)
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_EXT = Path(__file__).parent.parent.parent
if str(_EXT) not in sys.path:
    sys.path.insert(0, str(_EXT))

from splatforge.fields import GaussianField  # noqa: E402
from splatforge.remesh import remesh_step    # noqa: E402


# --------------------------------------------------------------------------- #
# Default weight presets
# --------------------------------------------------------------------------- #

DEFAULT_WEIGHTS: dict = {
    "Refined": {
        "w_iso":        1.0,
        "w_normal":     0.3,
        "w_edge":       0.02,
        "w_lap":        0.04,   # higher than ALPHA (0.005) — cotan is better conditioned
        "remesh_every": 40,
    },
    "Maximum": {
        "w_iso":        1.0,
        "w_normal":     0.3,
        "w_edge":       0.02,
        "w_lap":        0.04,
        "remesh_every": 40,
    },
}


# --------------------------------------------------------------------------- #
# Cached cotangent Laplacian
# --------------------------------------------------------------------------- #

def _build_cotan_cache(
    verts_np: np.ndarray,
    faces_np: np.ndarray,
    device: torch.device,
) -> tuple:
    """Precompute directed-edge cotangent weights from current geometry.

    Returns (src, dst, wts) — each shape (6F,) — as GPU tensors.

    Per-edge weight = 0.5 × cot(opposite angle), clamped to ≥0.
    In each triangle (v0, v1, v2):
      edge (0,1): opposite angle at v2   → weight = 0.5 × cot2
      edge (1,2): opposite angle at v0   → weight = 0.5 × cot0
      edge (2,0): opposite angle at v1   → weight = 0.5 × cot1
    Each undirected edge is stored as two directed edges (i→j and j→i).

    Called once at init and once after each remesh pass.
    """
    v = verts_np.astype(np.float64)
    f = faces_np

    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e01 = v1 - v0;  e02 = v2 - v0
    e10 = v0 - v1;  e12 = v2 - v1
    e20 = v0 - v2;  e21 = v1 - v2

    def _cot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        cross = np.linalg.norm(np.cross(a, b), axis=1).clip(1e-8)
        return np.maximum(0.0, (a * b).sum(1) / cross)

    cot0 = _cot(e01, e02)   # at v0, opposite edge (1,2)
    cot1 = _cot(e10, e12)   # at v1, opposite edge (0,2)
    cot2 = _cot(e20, e21)   # at v2, opposite edge (0,1)

    # 6 directed edges per face
    src = np.concatenate([f[:, 0], f[:, 1],   # edge 01
                          f[:, 1], f[:, 2],   # edge 12
                          f[:, 2], f[:, 0]])  # edge 20
    dst = np.concatenate([f[:, 1], f[:, 0],
                          f[:, 2], f[:, 1],
                          f[:, 0], f[:, 2]])
    wts = np.concatenate([0.5 * cot2, 0.5 * cot2,
                          0.5 * cot0, 0.5 * cot0,
                          0.5 * cot1, 0.5 * cot1]).astype(np.float32)

    return (
        torch.as_tensor(src, dtype=torch.long,    device=device),
        torch.as_tensor(dst, dtype=torch.long,    device=device),
        torch.as_tensor(wts, dtype=torch.float32, device=device),
    )


def _cotan_lap_loss(
    v: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    wts: torch.Tensor,
    nv: int,
) -> torch.Tensor:
    """Mean-squared cotangent Laplacian displacement, differentiable w.r.t. v.

    Uses cached (src, dst, wts) — topology is fixed between remesh passes.
    For vertices whose total cotan weight rounds to zero (all-obtuse patch),
    falls back to the uniform umbrella centroid.
    """
    wv = wts.unsqueeze(1)                                           # (E,1)
    nbr_sum = torch.zeros(nv, 3, device=v.device, dtype=v.dtype)
    wgt_sum = torch.zeros(nv, 1, device=v.device, dtype=v.dtype)

    nbr_sum.scatter_add_(0, src.unsqueeze(1).expand(-1, 3), wv * v[dst])
    wgt_sum.scatter_add_(0, src.unsqueeze(1), wv)

    zero_mask = wgt_sum < 1e-7                                      # (nv,1) bool

    if zero_mask.any():
        # Uniform fallback for degenerate vertices
        nbr_sum_u = torch.zeros(nv, 3, device=v.device, dtype=v.dtype)
        nbr_cnt   = torch.zeros(nv, 1, device=v.device, dtype=v.dtype)
        ones      = torch.ones(len(src), 1, device=v.device, dtype=v.dtype)
        nbr_sum_u.scatter_add_(0, src.unsqueeze(1).expand(-1, 3), v[dst])
        nbr_cnt.scatter_add_(0, src.unsqueeze(1), ones)

        c_cotan = nbr_sum / wgt_sum.clamp_min(1e-8)
        c_unif  = nbr_sum_u / nbr_cnt.clamp_min(1.0)
        centroid = torch.where(zero_mask, c_unif, c_cotan)
    else:
        centroid = nbr_sum / wgt_sum.clamp_min(1e-8)

    lap = v - centroid
    return (lap ** 2).sum(1).mean()


# --------------------------------------------------------------------------- #
# Shared geometry helpers (same as ALPHA)
# --------------------------------------------------------------------------- #

def _vertex_normals(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = torch.linalg.cross(v1 - v0, v2 - v0)
    vn = torch.zeros_like(v)
    vn.scatter_add_(0, f[:, 0:1].expand(-1, 3), fn)
    vn.scatter_add_(0, f[:, 1:2].expand(-1, 3), fn)
    vn.scatter_add_(0, f[:, 2:3].expand(-1, 3), fn)
    return F.normalize(vn, dim=1)


def _edge_var_loss(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return (v[f[:, 1]] - v[f[:, 0]]).norm(dim=1).var()


def _mean_edge_length_np(verts_np: np.ndarray, faces_np: np.ndarray) -> float:
    e01 = np.linalg.norm(verts_np[faces_np[:, 1]] - verts_np[faces_np[:, 0]], axis=1)
    return float(e01.mean()) if len(e01) > 0 else 1.0


# --------------------------------------------------------------------------- #
# Main optimiser
# --------------------------------------------------------------------------- #

def optimize_phase_a(
    verts: np.ndarray,
    faces: np.ndarray,
    field: "GaussianField",
    n_steps: int = 200,
    lr: float = 5e-4,
    weights: dict = None,
    device: torch.device = None,
    budget_s: float = 90.0,
    profile: bool = False,
) -> tuple:
    """Phase A geometry optimiser — HYBRID variant.

    ALPHA fidelity weights with a cached clamped cotangent Laplacian.
    Cotan weights are recomputed only after remesh passes (held fixed otherwise).

    Returns (verts_np float32, faces_np int64).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS["Refined"]

    w_iso        = float(weights.get("w_iso",        1.0))
    w_normal     = float(weights.get("w_normal",     0.3))
    w_edge       = float(weights.get("w_edge",       0.02))
    w_lap        = float(weights.get("w_lap",        0.04))
    remesh_every = int(weights.get("remesh_every",   40))

    if device is None:
        device = field.device

    tau = field.tau

    verts_np = np.array(verts, dtype=np.float32)
    faces_np = np.array(faces, dtype=np.int64)

    def _to_tensors(v_np, f_np):
        v = torch.tensor(v_np, dtype=torch.float32, device=device, requires_grad=True)
        f = torch.tensor(f_np, dtype=torch.long,    device=device)
        return v, f

    v, f = _to_tensors(verts_np, faces_np)
    src, dst, wts = _build_cotan_cache(verts_np, faces_np, device)

    opt = torch.optim.Adam([v], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.01)

    best_loss  = float("inf")
    best_verts = verts_np.copy()
    best_faces = faces_np.copy()

    t0 = time.time()
    _prof = {"iso": 0., "normal": 0., "edge": 0., "lap": 0.,
             "remesh": 0., "backward": 0.} if profile else None

    for step in range(n_steps):
        if time.time() - t0 > budget_s:
            break

        # ── remesh + cache refresh ──────────────────────────────────────── #
        if remesh_every > 0 and step > 0 and step % remesh_every == 0:
            _tr = time.perf_counter() if profile else 0.0
            v_np_cur = v.detach().cpu().numpy().astype(np.float32)
            f_np_cur = f.cpu().numpy().astype(np.int64)
            try:
                v_np_new, f_np_new = remesh_step(v_np_cur, f_np_cur, field)
                v_np_new = v_np_new.astype(np.float32)
                f_np_new = f_np_new.astype(np.int64)
                if len(v_np_new) > 0 and len(f_np_new) > 0:
                    v, f = _to_tensors(v_np_new, f_np_new)
                    # Rebuild cotan cache after topology change
                    src, dst, wts = _build_cotan_cache(v_np_new, f_np_new, device)
                    field.refresh_knn_cache()
                    remaining = n_steps - step
                    opt = torch.optim.Adam([v], lr=scheduler.get_last_lr()[0])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=max(remaining, 1), eta_min=lr * 0.01)
            except Exception:
                pass
            if profile:
                _prof["remesh"] += time.perf_counter() - _tr

        # ── forward pass ──────────────────────────────────────────────────── #
        opt.zero_grad()

        _t = time.perf_counter() if profile else 0.0
        rho   = field.density(v)
        L_iso = ((rho - tau) ** 2).mean()
        if profile: _prof["iso"] += time.perf_counter() - _t

        _t = time.perf_counter() if profile else 0.0
        vn    = _vertex_normals(v, f)
        g     = field.density_grad(v)
        L_normal = (1.0 - (vn * F.normalize(g, dim=1)).sum(1).abs()).mean()
        if profile: _prof["normal"] += time.perf_counter() - _t

        _t = time.perf_counter() if profile else 0.0
        L_edge = _edge_var_loss(v, f)
        if profile: _prof["edge"] += time.perf_counter() - _t

        _t = time.perf_counter() if profile else 0.0
        nv = v.shape[0]
        L_lap = _cotan_lap_loss(v, src, dst, wts, nv)
        if profile: _prof["lap"] += time.perf_counter() - _t

        loss = w_iso * L_iso + w_normal * L_normal + w_edge * L_edge + w_lap * L_lap

        if loss.isnan():
            break

        with torch.no_grad():
            v_prev   = v.detach().clone()
            mel      = _mean_edge_length_np(
                v_prev.cpu().numpy().astype(np.float32),
                f.cpu().numpy().astype(np.int64))
            max_step = 0.5 * mel

        _t = time.perf_counter() if profile else 0.0
        loss.backward()
        opt.step()
        scheduler.step()
        if profile: _prof["backward"] += time.perf_counter() - _t

        # step clamping
        with torch.no_grad():
            delta = v.data - v_prev
            scale = (max_step / delta.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp_max(1.0)
            v.data.copy_(v_prev + delta * scale)

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss  = loss_val
            best_verts = v.detach().cpu().numpy().astype(np.float32)
            best_faces = f.cpu().numpy().astype(np.int64)

    if profile and _prof is not None:
        total = sum(_prof.values())
        print(f"[HYBRID profile] total={total:.1f}s  "
              + "  ".join(f"{k}={v:.1f}s({v/total*100:.0f}%)" for k, v in _prof.items()),
              flush=True)

    return best_verts, best_faces

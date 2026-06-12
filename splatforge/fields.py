"""
Analytic Gaussian density field ρ(x), gradient ∇ρ(x), and color c(x).

Gaussian parameters (centers, covariances, opacity, rgb) are frozen at
construction. Only the query positions x passed to density() / density_grad()
are differentiable — gradients flow through the quadratic form
  d = x − μᵢ,  quad = dᵀ Σ⁻¹ d
and then exp(−½ quad).

KNN lookup is done on x.detach() (frozen connectivity per step), which is the
standard frozen-neighbourhood assumption used in all mesh-based differentiable
rendering. Steps are small enough (< 0.5× local edge length) that neighbours
don't change between optimizer iterations.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

_EXT = Path(__file__).parent.parent
if str(_EXT) not in sys.path:
    sys.path.insert(0, str(_EXT))
from splat_mesh import _inverse_covariance, _otsu_level


def _median_spacing(xyz_np: np.ndarray) -> float:
    if len(xyz_np) < 2:
        return 1.0
    tree = cKDTree(xyz_np)
    sub = xyz_np[np.random.default_rng(0).choice(
        len(xyz_np), min(20_000, len(xyz_np)), replace=False)]
    return float(np.median(tree.query(sub, k=2)[0][:, 1]))


class GaussianField:
    """Frozen Gaussian mixture field.

    ρ(x) = Σᵢ opᵢ · exp(−½ (x−μᵢ)ᵀ Σᵢ⁻¹ (x−μᵢ))

    gaussians: dict with float32 tensors
      xyz      (N,3)  – Gaussian centers, splat/Z-up frame
      opacity  (N,)   – values in [0,1]
      scale    (N,3)  – linear std (not log)
      quat     (N,4)  – wxyz normalized
      rgb      (N,3)  – DC color in [0,1]
    """

    _K = 48           # nearest Gaussians per query
    _CHUNK = 16_384   # query points per GPU batch (limits VRAM, NOT CPU KNN)
    _KNN_REFRESH = 10 # recompute KNN every N density() / density_grad() calls

    def __init__(self, gaussians: dict, device: torch.device, tau_res: int = 64,
                 surface_pts: np.ndarray = None):
        xyz     = gaussians["xyz"].detach().float()
        opacity = gaussians["opacity"].detach().reshape(-1).float()
        scale   = gaussians["scale"].detach().float()
        rgb     = gaussians["rgb"].detach().float()

        # Normalize quats: exported values are raw network outputs (non-unit).
        # _inverse_covariance also normalizes internally, but we assert/normalize
        # here so the invariant is explicit and callers can rely on it.
        quat = gaussians["quat"].detach().float()
        norms = quat.norm(dim=1, keepdim=True).clamp_min(1e-6)
        max_dev = float((norms - 1.0).abs().max())
        if max_dev > 0.05:
            print(f"[GaussianField] quat norms off-unit "
                  f"(median={float(norms.median()):.3f}, max_dev={max_dev:.3f}) — normalizing",
                  flush=True)
        quat = quat / norms

        centers_np = xyz.cpu().numpy().astype(np.float64)
        # 0.9 × median spacing — MUST match splat_mesh.py's scale_floor_frac=0.9.
        # The reference surface was extracted from a density built with this floor;
        # a smaller floor (e.g. 0.25) yields a sharper field whose iso-surface sits
        # INSIDE the reference, so the optimizer pulls vertices off-target.
        floor = 0.9 * _median_spacing(centers_np)

        self.sinv    = _inverse_covariance(scale.clamp_min(floor), quat).to(device)
        self.centers = xyz.to(device)
        self.opacity = opacity.to(device)
        self.rgb     = rgb.to(device)
        self.device  = device
        self.K       = min(self._K, len(centers_np))

        self._tree       = cKDTree(centers_np)
        self._centers_np = centers_np.astype(np.float32)

        # KNN cache: computed once for all Q query pts, refreshed every _KNN_REFRESH calls
        self._knn_cache:     np.ndarray | None = None
        self._knn_call_cnt:  int = 0

        # tau: prefer direct surface alignment — the median density on the known
        # reference/start surface IS the iso-level that reproduces it (absorbs the
        # grid-resolution, Otsu-binning and Taubin-smoothing differences vs
        # splat_mesh.py). Falls back to the coarse-grid Otsu estimate.
        if surface_pts is not None and len(surface_pts) > 0:
            self.tau = self._tau_from_surface(surface_pts)
        else:
            self.tau = self._estimate_tau(tau_res)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _knn_idx(self, x_np: np.ndarray) -> np.ndarray:
        """KNN for ALL query pts at once (one CPU query), with per-call caching.
        Returns (Q, K) int64. Cache is invalidated by refresh_knn_cache()."""
        self._knn_call_cnt += 1
        if (self._knn_cache is None
                or len(x_np) != len(self._knn_cache)
                or self._knn_call_cnt % self._KNN_REFRESH == 0):
            _, idx = self._tree.query(x_np, k=self.K, workers=-1)
            if self.K == 1:
                idx = idx[:, None]
            self._knn_cache = idx.astype(np.int64)
        return self._knn_cache

    def refresh_knn_cache(self) -> None:
        """Force KNN recomputation on the next density/density_grad call.
        Call this after remesh (vertex set changes)."""
        self._knn_cache = None

    def _tau_from_surface(self, pts: np.ndarray) -> float:
        """Median density over points known to lie on the target surface.
        Uses a subsample for speed; resets the KNN cache afterwards so the
        first optimizer call doesn't reuse the surface-pts cache."""
        pts = np.asarray(pts, np.float32)
        if len(pts) > 50_000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 50_000, replace=False)]
        pts_t = torch.as_tensor(pts, device=self.device)
        with torch.no_grad():
            rho = self.density(pts_t)
        self.refresh_knn_cache()
        return float(rho.median())

    def _estimate_tau(self, res: int) -> float:
        """Otsu threshold via coarse grid density probe (res³ points)."""
        lo = self._centers_np.min(0) - 0.1
        hi = self._centers_np.max(0) + 0.1
        axes = [np.linspace(lo[i], hi[i], res, dtype=np.float32) for i in range(3)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        pts_t = torch.as_tensor(pts, device=self.device)
        with torch.no_grad():
            rho = self.density(pts_t).cpu().numpy()
        vmax = float(rho.max())
        occ = rho[rho > vmax * 1e-3]
        if len(occ) == 0:
            return vmax * 0.3
        return float(_otsu_level(occ) * 0.4)  # same level_bias as splat_mesh.py

    # ------------------------------------------------------------------ #
    # Public field evaluations
    # ------------------------------------------------------------------ #

    def density(self, x: torch.Tensor) -> torch.Tensor:
        """ρ(x), differentiable w.r.t. x.  x: (Q,3) float32 → (Q,) float32.

        KNN is computed once for all Q pts and cached for _KNN_REFRESH calls
        (vertices move slowly, so stale neighbours are acceptable for ~10 steps).
        """
        x_np = x.detach().cpu().numpy().astype(np.float64)
        all_idx = self._knn_idx(x_np)                        # (Q,K) — one CPU query
        chunks = []
        for s in range(0, len(x), self._CHUNK):
            e = s + self._CHUNK
            idx_t = torch.as_tensor(all_idx[s:e], device=self.device, dtype=torch.long)
            d      = x[s:e, None, :] - self.centers[idx_t]  # (T,K,3) diff w.r.t. x
            sinv_k = self.sinv[idx_t]                        # (T,K,3,3) frozen
            quad   = torch.einsum("tki,tkij,tkj->tk", d, sinv_k, d)
            w      = self.opacity[idx_t] * torch.exp(-0.5 * quad.clamp_max(20.0))
            chunks.append(w.sum(1))
        return torch.cat(chunks)

    def density_grad(self, x: torch.Tensor) -> torch.Tensor:
        """∇ρ(x), analytical, detached. x: (Q,3) → (Q,3) float32.
        Shares the KNN cache with density() — call them in the same step to reuse."""
        out_list = []
        x_np = x.detach().cpu().numpy().astype(np.float64)
        all_idx = self._knn_idx(x_np)
        with torch.no_grad():
            for s in range(0, len(x), self._CHUNK):
                e = s + self._CHUNK
                xc     = x[s:e].detach()
                idx_t  = torch.as_tensor(all_idx[s:e], device=self.device, dtype=torch.long)
                d      = xc[:, None, :] - self.centers[idx_t]
                sinv_k = self.sinv[idx_t]
                quad   = torch.einsum("tki,tkij,tkj->tk", d, sinv_k, d)
                w      = self.opacity[idx_t] * torch.exp(-0.5 * quad.clamp_max(20.0))
                Sd     = torch.einsum("tkij,tkj->tki", sinv_k, d)
                out_list.append(-(w[:, :, None] * Sd).sum(1))
        return torch.cat(out_list)

    def color(self, x: torch.Tensor) -> torch.Tensor:
        """c(x) = Σ wᵢ rgbᵢ / Σ wᵢ, detached. x: (Q,3) → (Q,3) float32."""
        out_list = []
        x_np = x.detach().cpu().numpy().astype(np.float64)
        all_idx = self._knn_idx(x_np)
        with torch.no_grad():
            for s in range(0, len(x), self._CHUNK):
                e      = s + self._CHUNK
                idx_t  = torch.as_tensor(all_idx[s:e], device=self.device, dtype=torch.long)
                d      = x[s:e].detach()[:, None, :] - self.centers[idx_t]
                sinv_k = self.sinv[idx_t]
                quad   = torch.einsum("tki,tkij,tkj->tk", d, sinv_k, d)
                w      = self.opacity[idx_t] * torch.exp(-0.5 * quad.clamp_max(20.0))
                sw     = w.sum(1, keepdim=True).clamp_min(1e-8)
                out_list.append((w[:, :, None] * self.rgb[idx_t]).sum(1) / sw)
        return torch.cat(out_list)


if __name__ == "__main__":
    # Unit test: non-unit quats must produce density identical to their normalized version.
    torch.manual_seed(0)
    dev = torch.device("cpu")
    N = 32
    xyz     = torch.randn(N, 3)
    opacity = torch.rand(N).clamp_min(0.01)
    scale   = torch.rand(N, 3).clamp(0.01, 0.5)
    rgb     = torch.rand(N, 3)
    quat_unit = torch.randn(N, 4)
    quat_unit = quat_unit / quat_unit.norm(dim=1, keepdim=True)
    # Scale each quat by a random positive factor in [0.1, 3.0] — mimics raw network output
    quat_raw = quat_unit * torch.rand(N, 1).clamp(0.1, 3.0)

    g = lambda q: {"xyz": xyz, "opacity": opacity, "scale": scale, "quat": q, "rgb": rgb}
    f_unit = GaussianField(g(quat_unit), dev)
    f_raw  = GaussianField(g(quat_raw),  dev)

    x = torch.randn(16, 3)
    rho_unit = f_unit.density(x)
    rho_raw  = f_raw.density(x)
    max_diff = float((rho_unit - rho_raw).abs().max())
    assert max_diff < 1e-5, f"FAIL: max_diff={max_diff:.2e}"
    print(f"quat normalization test PASSED  max_diff={max_diff:.2e}")

"""
Pure-torch 3DGS splat renderer (reference images for the Phase B gate).

Standard EWA splatting pipeline, no custom CUDA:
  - project centers through a pinhole camera
  - 2D covariance via the perspective Jacobian:  S' = J W S Wt Jt  (+0.3 px low-pass)
  - depth-sort, then composite front-to-back in chunks:
      within a chunk (default 4096 gaussians) the combined transmittance is exact
      ( prod(1-a) = exp(sum log(1-a)) accumulated with scatter_add ) and the chunk
      colour is the alpha-weighted mean — i.e. ordering is ignored only INSIDE a
      chunk. With depth-sorted chunks this approximation error is negligible and
      everything stays vectorized.
  - rasterization uses radius buckets (same pattern as splat_mesh._splat_density):
    each gaussian writes a (2k+1)^2 window sized to its own 3 sigma, capped.

Output: (rgb (H,W,3) float32 in [0,1], alpha (H,W) float32). Background = 0.
Not differentiable (gaussians are frozen reference); kept eval-only.
"""
import numpy as np
import torch


def _covariance_3d(scale: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Sigma = R diag(s^2) Rt.  scale (N,3) linear std, quat (N,4) wxyz (any norm)."""
    q = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)
    return torch.einsum("nij,nj,nkj->nik", R, scale ** 2, R)


@torch.no_grad()
def render_splats(xyz, opacity, scale, quat, rgb, cam: dict,
                  chunk: int = 4096, max_radius_px: int = 24,
                  alpha_thresh: float = 1.0 / 255.0,
                  return_depth: bool = False):
    """Render frozen gaussians from one camera. All tensors on the same device.

    cam: {R (3,3), t (3,), fx, fy, cx, cy, res} — torch tensors for R/t.
    Returns (rgb (H,W,3), alpha (H,W)) float32 on the same device.
    return_depth: additionally return expected depth (H,W) — alpha-weighted mean
    camera-z (standard 3DGS depth), 0 where no coverage. The splat depth is
    SMOOTH across the surface, which makes it the reference for judging
    crevasse-type geometry noise on the mesh side.
    """
    dev = xyz.device
    H = W = int(cam["res"])
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

    # ── camera space + projection ─────────────────────────────────────────
    p_cam = xyz @ cam["R"].T + cam["t"]                     # (N,3)
    z = p_cam[:, 2]
    in_front = z > 1e-4
    px = fx * p_cam[:, 0] / z.clamp_min(1e-6) + cx
    py = fy * p_cam[:, 1] / z.clamp_min(1e-6) + cy

    # ── 2D covariance: J W S Wt Jt ────────────────────────────────────────
    S = _covariance_3d(scale, quat)                          # (N,3,3) world
    Sc = torch.einsum("ij,njk,lk->nil", cam["R"], S, cam["R"])  # camera-frame
    iz = 1.0 / z.clamp_min(1e-6)
    J = torch.zeros(len(xyz), 2, 3, device=dev)
    J[:, 0, 0] = fx * iz
    J[:, 0, 2] = -fx * p_cam[:, 0] * iz * iz
    J[:, 1, 1] = fy * iz
    J[:, 1, 2] = -fy * p_cam[:, 1] * iz * iz
    S2 = torch.einsum("nij,njk,nlk->nil", J, Sc, J)          # (N,2,2)
    S2[:, 0, 0] += 0.3                                       # 3DGS low-pass
    S2[:, 1, 1] += 0.3

    det = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] * S2[:, 1, 0]
    valid = in_front & (det > 1e-12)
    # eigenvalue upper bound -> 3 sigma pixel radius
    mid = 0.5 * (S2[:, 0, 0] + S2[:, 1, 1])
    lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.0))
    radius = torch.ceil(3.0 * torch.sqrt(lam.clamp_min(0.0)))
    on_screen = (px + radius > 0) & (px - radius < W) & (py + radius > 0) & (py - radius < H)
    valid &= on_screen & (radius >= 1)

    idx_v = valid.nonzero(as_tuple=True)[0]
    if idx_v.numel() == 0:
        empty = (torch.zeros(H, W, 3, device=dev), torch.zeros(H, W, device=dev))
        return empty + (torch.zeros(H, W, device=dev),) if return_depth else empty

    # inverse 2D covariance (conic)
    inv_det = 1.0 / det[idx_v]
    a = S2[idx_v, 1, 1] * inv_det
    b = -S2[idx_v, 0, 1] * inv_det
    c = S2[idx_v, 0, 0] * inv_det

    # ── depth sort (front to back) ────────────────────────────────────────
    order = torch.argsort(z[idx_v])
    idx_v = idx_v[order]
    a, b, c = a[order], b[order], c[order]
    gx, gy = px[idx_v], py[idx_v]
    g_op = opacity[idx_v].clamp(0.0, 0.999)
    g_rgb = rgb[idx_v]
    g_z = z[idx_v]
    g_rad = radius[idx_v].clamp_max(float(max_radius_px)).long()

    # ── composite in depth chunks ─────────────────────────────────────────
    C = torch.zeros(H * W, 3, device=dev)
    D = torch.zeros(H * W, device=dev) if return_depth else None
    T = torch.ones(H * W, device=dev)                       # transmittance

    n = idx_v.numel()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        logt = torch.zeros(H * W, device=dev)               # sum log(1-alpha)
        wc = torch.zeros(H * W, 3, device=dev)              # sum alpha*color
        ws = torch.zeros(H * W, device=dev)                 # sum alpha
        wz = torch.zeros(H * W, device=dev) if return_depth else None

        rad_c = g_rad[s:e]
        for k in torch.unique(rad_c).tolist():
            sel = (rad_c == k).nonzero(as_tuple=True)[0] + s
            rng = torch.arange(-k, k + 1, device=dev, dtype=torch.float32)
            oy, ox = torch.meshgrid(rng, rng, indexing="ij")
            off = torch.stack([ox.reshape(-1), oy.reshape(-1)], 1)   # (M,2)

            cxi = gx[sel].round()[:, None] + off[None, :, 0]         # (B,M)
            cyi = gy[sel].round()[:, None] + off[None, :, 1]
            dx = cxi - gx[sel][:, None]
            dy = cyi - gy[sel][:, None]
            quad = (a[sel][:, None] * dx * dx
                    + 2.0 * b[sel][:, None] * dx * dy
                    + c[sel][:, None] * dy * dy)
            alpha = g_op[sel][:, None] * torch.exp(-0.5 * quad)
            alpha = torch.where(quad < 9.0, alpha, torch.zeros_like(alpha))
            inside = (cxi >= 0) & (cxi < W) & (cyi >= 0) & (cyi < H) & (alpha > alpha_thresh)

            flat = (cyi.clamp(0, H - 1).long() * W + cxi.clamp(0, W - 1).long()).reshape(-1)
            al = torch.where(inside, alpha, torch.zeros_like(alpha)).reshape(-1)
            keep = al > 0
            flat, al = flat[keep], al[keep]
            col = g_rgb[sel][:, None, :].expand(-1, off.shape[0], -1).reshape(-1, 3)[keep]

            logt.scatter_add_(0, flat, torch.log1p(-al.clamp_max(0.999)))
            ws.scatter_add_(0, flat, al)
            wc.scatter_add_(0, flat[:, None].expand(-1, 3), al[:, None] * col)
            if return_depth:
                zg = g_z[sel][:, None].expand(-1, off.shape[0]).reshape(-1)[keep]
                wz.scatter_add_(0, flat, al * zg)

        A = 1.0 - torch.exp(logt)                           # chunk opacity, exact
        chunk_col = wc / ws.clamp_min(1e-8)[:, None]        # alpha-weighted mean
        C += (T * A)[:, None] * chunk_col
        if return_depth:
            D += (T * A) * (wz / ws.clamp_min(1e-8))
        T *= 1.0 - A
        if float(T.max()) < 1e-3:                           # fully opaque everywhere
            break

    alpha = (1.0 - T)
    if return_depth:
        depth = torch.where(alpha > 1e-3, D / alpha.clamp_min(1e-8),
                            torch.zeros_like(D))
        return (C.reshape(H, W, 3).clamp(0, 1), alpha.reshape(H, W),
                depth.reshape(H, W))
    return C.reshape(H, W, 3).clamp(0, 1), alpha.reshape(H, W)

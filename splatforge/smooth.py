"""
Detail-preserving mesh denoising: bilateral normal filtering + vertex update.

Targets the "crevasse" artifact of Surface-Nets-from-gaussians surfaces: the
density field dips between neighbouring gaussians, so the iso-surface carries
high-frequency valleys at the inter-gaussian spacing wavelength. Plain Taubin
smoothing attacks ALL frequencies (melts ears/edges before crevasses are gone);
bilateral normal filtering smooths normals only where neighbouring normals
already roughly agree, so spacing-scale noise is flattened while genuine sharp
features (normal discontinuities) survive.

Reference algorithm: Zheng et al. 2011, "Bilateral Normal Filtering for Mesh
Denoising" (local scheme) + Sun et al. vertex update. Pure torch, O(F) via
edge-adjacency scatter ops.
"""
import numpy as np
import torch


def _face_geometry(v: torch.Tensor, f: torch.Tensor):
    """Face normals (unit), centroids, areas."""
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = torch.linalg.cross(p1 - p0, p2 - p0)
    area = 0.5 * cr.norm(dim=1)
    n = torch.nn.functional.normalize(cr, dim=1)
    c = (p0 + p1 + p2) / 3.0
    return n, c, area


def _face_adjacency(faces_np: np.ndarray) -> np.ndarray:
    """(M,2) pairs of face indices sharing an edge (each pair once)."""
    F = len(faces_np)
    edges = np.concatenate([
        np.sort(faces_np[:, [0, 1]], axis=1),
        np.sort(faces_np[:, [1, 2]], axis=1),
        np.sort(faces_np[:, [2, 0]], axis=1),
    ], axis=0)
    face_of = np.tile(np.arange(F, dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, face_of = edges[order], face_of[order]
    same = (edges[1:] == edges[:-1]).all(axis=1)
    return np.stack([face_of[:-1][same], face_of[1:][same]], axis=1)


def bilateral_smooth(verts_np: np.ndarray, faces_np: np.ndarray,
                     normal_iters: int = 8, vertex_iters: int = 15,
                     sigma_r: float = 0.35, device=None) -> np.ndarray:
    """Denoise vertex positions; topology unchanged. Returns new verts (V,3) f32.

    sigma_r: range sigma on normal difference norm (0.2 aggressive-preserve /
             0.5 closer to plain smoothing). sigma_s is auto-set to 2x the mean
             adjacent-centroid distance (the crevasse wavelength scale).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v = torch.as_tensor(np.asarray(verts_np, np.float32), device=device)
    f = torch.as_tensor(np.asarray(faces_np, np.int64), device=device)
    adj = torch.as_tensor(_face_adjacency(np.asarray(faces_np, np.int64)),
                          device=device)
    ai, bj = adj[:, 0], adj[:, 1]

    n, c, area = _face_geometry(v, f)
    sigma_s = 2.0 * float((c[ai] - c[bj]).norm(dim=1).mean())

    # ── bilateral normal filtering ────────────────────────────────────────
    for _ in range(normal_iters):
        d_c = (c[ai] - c[bj]).norm(dim=1)
        d_n = (n[ai] - n[bj]).norm(dim=1)
        w = (area[bj] * torch.exp(-0.5 * (d_c / sigma_s) ** 2)
                      * torch.exp(-0.5 * (d_n / sigma_r) ** 2))
        w_rev = (area[ai] * torch.exp(-0.5 * (d_c / sigma_s) ** 2)
                          * torch.exp(-0.5 * (d_n / sigma_r) ** 2))

        acc = n * area[:, None]                     # self term
        acc.scatter_add_(0, ai[:, None].expand(-1, 3), w[:, None] * n[bj])
        acc.scatter_add_(0, bj[:, None].expand(-1, 3), w_rev[:, None] * n[ai])
        n = torch.nn.functional.normalize(acc, dim=1)

    # ── vertex update to match filtered normals (Sun et al.) ─────────────
    nv = len(v)
    deg = torch.zeros(nv, 1, device=device)
    ones = torch.ones(len(f), 1, device=device)
    for k in range(3):
        deg.scatter_add_(0, f[:, k:k + 1], ones)
    deg = deg.clamp_min(1.0)

    for _ in range(vertex_iters):
        _, c, _ = _face_geometry(v, f)
        # per-face correction projected on the filtered normal, gathered per vertex
        upd = torch.zeros_like(v)
        for k in range(3):
            corr = n * ((c - v[f[:, k]]) * n).sum(1, keepdim=True)
            upd.scatter_add_(0, f[:, k:k + 1].expand(-1, 3), corr)
        v = v + upd / deg

    return v.cpu().numpy().astype(np.float32)

"""
Baked-texture path for the TripoSplat mesh.

Instead of trilinearly sampling a low-res fp16 voxel colour volume into ~1 vertex
colour per vertex (the quality ceiling of the vertex-colour path), this UV-unwraps
the final mesh with xatlas and evaluates each texel's colour DIRECTLY from the
Gaussians: per texel, the K nearest Gaussians are weighted by
    w_i = opacity_i * exp(-0.5 * dᵀ Σ⁻¹ d)
(the exact splat density kernel, Σ⁻¹ reused from splat_mesh._inverse_covariance) and
the colour is Σ w_i c_i / Σ w_i.

Colour-space rule is identical to the vertex path: raw display colours, NO sRGB pass.
The geometry `scale_floor_frac` floor is deliberately NOT applied here (flooring blurs
colour); only a minimal floor (0.25 × median inter-Gaussian spacing) is used so no
texel goes dead. The mesh is returned in the SAME (splat / Z-up) space it came in;
the caller applies the glTF Y/Z flip, exactly as for the vertex path.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from splat_mesh import _inverse_covariance


class UnwrapError(RuntimeError):
    """UV unwrap failed (xatlas crashed or timed out in the worker subprocess)."""


def _unwrap(verts, faces, timeout=90):
    """UV-unwrap via xatlas, isolated in a subprocess so a native segfault becomes a
    catchable error instead of killing the Modly backend. Returns (vmapping, indices,
    uvs). Caller must keep the mesh under xatlas's Windows crash threshold (~120k faces
    irregular) — the generator decimates to ~80k before baking."""
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unwrap_worker.py")
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.npz")
        outp = os.path.join(td, "out.npz")
        np.savez(inp, verts=np.ascontiguousarray(verts, np.float32),
                 faces=np.ascontiguousarray(faces, np.uint32))
        try:
            r = subprocess.run([sys.executable, worker, inp, outp],
                               timeout=timeout, capture_output=True)
        except subprocess.TimeoutExpired:
            raise UnwrapError(f"xatlas unwrap timed out (>{timeout}s)")
        if r.returncode != 0 or not os.path.exists(outp):
            err = (r.stderr or b"").decode("utf-8", "ignore")[-300:]
            raise UnwrapError(f"xatlas unwrap failed (exit {r.returncode}) {err}")
        # Materialize the arrays and close the file handle before the TemporaryDirectory
        # is removed — Windows refuses to delete a still-open .npz (WinError 32).
        with np.load(outp) as d:
            return d["vmapping"].copy(), d["indices"].copy(), d["uvs"].copy()


def _median_spacing(centers_np):
    """Median nearest-neighbour distance of the Gaussian cloud (CPU, subsampled)."""
    n = len(centers_np)
    if n < 2:
        return 1.0
    tree = cKDTree(centers_np)
    sub = centers_np[np.random.default_rng(0).choice(n, min(20000, n), replace=False)]
    return float(np.median(tree.query(sub, k=2)[0][:, 1]))


def _rasterize_uv(uvs, faces, verts3d, res, chunk=100000):
    """Rasterize each triangle's UV footprint into texels (vectorized, ragged-grid
    expansion per triangle chunk). For every covered texel, barycentric-interpolate
    the 3D surface position. Returns (texel_flat_idx int64 (P,), positions f32 (P,3)),
    deduped to one entry per texel."""
    P = uvs.astype(np.float32) * res  # pixel space
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    flats, poss = [], []
    for s in range(0, len(faces), chunk):
        i0, i1, i2 = f0[s:s + chunk], f1[s:s + chunk], f2[s:s + chunk]
        p0, p1, p2 = P[i0], P[i1], P[i2]            # (T,2)
        v0, v1, v2 = verts3d[i0], verts3d[i1], verts3d[i2]  # (T,3)

        umin = np.clip(np.floor(np.minimum(np.minimum(p0[:, 0], p1[:, 0]), p2[:, 0])), 0, res - 1).astype(np.int64)
        umax = np.clip(np.ceil(np.maximum(np.maximum(p0[:, 0], p1[:, 0]), p2[:, 0])), 0, res - 1).astype(np.int64)
        vmin = np.clip(np.floor(np.minimum(np.minimum(p0[:, 1], p1[:, 1]), p2[:, 1])), 0, res - 1).astype(np.int64)
        vmax = np.clip(np.ceil(np.maximum(np.maximum(p0[:, 1], p1[:, 1]), p2[:, 1])), 0, res - 1).astype(np.int64)
        w = umax - umin + 1
        h = vmax - vmin + 1
        area = w * h
        total = int(area.sum())
        if total == 0:
            continue

        tri = np.repeat(np.arange(len(p0)), area)
        rel = np.arange(total) - np.repeat(np.cumsum(area) - area, area)
        w_rep = w[tri]
        iu = umin[tri] + (rel % w_rep)
        iv = vmin[tri] + (rel // w_rep)
        cx = iu.astype(np.float32) + 0.5
        cy = iv.astype(np.float32) + 0.5

        ax, ay = p0[tri, 0], p0[tri, 1]
        bx, by = p1[tri, 0], p1[tri, 1]
        gx, gy = p2[tri, 0], p2[tri, 1]
        denom = (by - gy) * (ax - gx) + (gx - bx) * (ay - gy)
        safe = np.abs(denom) > 1e-12
        denom = np.where(safe, denom, 1.0)
        l0 = ((by - gy) * (cx - gx) + (gx - bx) * (cy - gy)) / denom
        l1 = ((gy - ay) * (cx - gx) + (ax - gx) * (cy - gy)) / denom
        l2 = 1.0 - l0 - l1
        eps = -1e-4
        inside = safe & (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
        if not inside.any():
            continue

        t = tri[inside]
        pos = (l0[inside, None] * v0[t] + l1[inside, None] * v1[t] + l2[inside, None] * v2[t])
        flats.append(iv[inside] * res + iu[inside])
        poss.append(pos.astype(np.float32))

    if not flats:
        return np.zeros(0, np.int64), np.zeros((0, 3), np.float32)
    flat = np.concatenate(flats)
    pos = np.concatenate(poss)
    uniq, first = np.unique(flat, return_index=True)
    return uniq, pos[first]


def _eval_color(positions, g, device, chunk=262144):
    """Exact Gaussian colour per texel: K=32 nearest, weighted by the splat density
    kernel. Minimal scale floor only; nearest-Gaussian fallback when Σw underflows."""
    xyz = g["xyz"].detach().float()
    opacity = g["opacity"].detach().reshape(-1).float()
    scale = g["scale"].detach().float()
    quat = g["quat"].detach().float()
    rgb = g["rgb"].detach().float()

    centers_np = xyz.cpu().numpy().astype(np.float64)
    tree = cKDTree(centers_np)
    n = len(centers_np)
    floor = 0.25 * _median_spacing(centers_np)

    sinv = _inverse_covariance(torch.clamp(scale, min=floor), quat).to(device)  # (N,3,3)
    centers = xyz.to(device)
    op = opacity.to(device)
    col = rgb.to(device)
    K = min(32, n)

    out = np.empty((len(positions), 3), np.float32)
    for s in range(0, len(positions), chunk):
        pc = positions[s:s + chunk].astype(np.float64)
        dist, idx = tree.query(pc, k=K, workers=-1)
        if K == 1:
            idx = idx[:, None]
        idx_t = torch.as_tensor(np.ascontiguousarray(idx), device=device, dtype=torch.long)
        p_t = torch.as_tensor(pc, device=device, dtype=torch.float32)
        d = p_t[:, None, :] - centers[idx_t]                 # (T,K,3)
        quad = torch.einsum("tki,tkij,tkj->tk", d, sinv[idx_t], d)
        w = op[idx_t] * torch.exp(-0.5 * quad)               # (T,K)
        sw = w.sum(1)
        c = (w[..., None] * col[idx_t]).sum(1) / sw[..., None].clamp_min(1e-8)
        bad = sw < 1e-8
        if bool(bad.any()):
            c[bad] = col[idx_t[:, 0]][bad]
        out[s:s + len(pc)] = c.detach().cpu().numpy()
    return out


def _apply_vivid(col):
    """Luminance-scaled saturation pre-boost (copied from splat_mesh.py)."""
    luma = (col @ np.array([0.299, 0.587, 0.114], np.float32))[:, None]
    amt = np.float32(0.5) * np.clip(luma / np.float32(0.4), 0.0, 1.0)
    return np.clip(luma + (col - luma) * (1.0 + amt), 0.0, 1.0).astype(np.float32)


def bake_texture(verts, faces, gaussians, resolution, device, vivid_color):
    """UV-unwrap `verts/faces` and bake a texture sampled directly from `gaussians`.

    gaussians: dict of torch tensors {xyz (N,3), opacity (N,), scale (N,3) linear std,
    quat (N,4) wxyz, rgb (N,3) display 0..1}. Returns (new_verts f32 (M,3, splat space),
    new_faces i64 (F,3), uvs f32 (M,2) in [0,1], texture_rgb_uint8 (res,res,3)).
    Raises UnwrapError if the UV unwrap fails."""
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    vmapping, indices, uvs = _unwrap(verts, faces)

    new_verts = verts[vmapping].astype(np.float32)
    new_faces = indices.astype(np.int64)
    uvs = np.clip(uvs.astype(np.float32), 0.0, 1.0)

    res = int(resolution)
    texel_idx, positions = _rasterize_uv(uvs, new_faces, new_verts, res)

    tex = np.zeros((res, res, 3), np.float32)
    mask = np.zeros((res, res), dtype=bool)
    if len(texel_idx) > 0:
        colors = _eval_color(positions, gaussians, device)
        if vivid_color:
            colors = _apply_vivid(colors)
        iu = (texel_idx % res).astype(np.int64)
        iv = (texel_idx // res).astype(np.int64)
        row = res - 1 - iv            # glTF UV origin: v up, image row down
        tex[row, iu] = colors
        mask[row, iu] = True

    # Gutter: nearest-fill every empty texel so bilinear/mip sampling never bleeds the
    # black background into UV seams.
    if mask.any() and not mask.all():
        inds = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        tex = tex[tuple(inds)]

    tex_u8 = (np.clip(tex, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return new_verts, new_faces, uvs, tex_u8

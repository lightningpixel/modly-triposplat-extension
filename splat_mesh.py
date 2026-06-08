"""
Gaussian-splat -> mesh, ported from ComfyUI's native `SplatToMesh`
(comfy_extras/nodes_gaussian_splat.py, VAST-AI / Comfy-Org).

Pipeline: each Gaussian is rasterized as its oriented 3-sigma covariance disk into a
density grid (+ a co-splatted colour volume), the iso-surface is extracted with
Surface Nets (dual contouring) at an Otsu-picked level, floaters/inner shells are
dropped, the surface is Taubin-smoothed, and vertex colours are trilinearly sampled
from the colour volume. This respects each splat's opacity (weight) and scale+rotation
(extent), so the surface fills solidly instead of pitting.

Colour is the trilinearly-sampled splat DC colour, written as raw display vertex
colours (NO sRGB->linear pass: Modly's viewer shows COLOR_0 as-is and does not
re-encode, so converting would only darken the mesh). `vivid_color` optionally
pre-boosts saturation. The glTF Y/Z flip is handled by the caller.
"""
import numpy as np
import torch
from scipy.ndimage import map_coordinates, minimum as _ndi_min, maximum as _ndi_max
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

_C0 = 0.28209479177387814  # SH band-0 DC -> base RGB


def _inverse_covariance(scale, quat):
    """Per-splat Sigma^-1 = R diag(1/s^2) R^T. scale (N,3) linear std, quat (N,4) wxyz."""
    q = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)
    inv_s2 = 1.0 / scale.clamp_min(1e-8) ** 2
    return torch.einsum("nij,nj,nkj->nik", R, inv_s2, R)


def _splat_density(xyz, opacity, scale, quat, rgb, res, kernel, device,
                   color_sharpen=2.0, chunk=4096, col_dtype=torch.float16):
    """Density grid + co-splatted colour volume. Each Gaussian uses a voxel window
    sized to its own 3-sigma (capped at `kernel`). Returns (density, colour-num,
    colour-norm, origin, voxel)."""
    pad = 4.0 * scale.median()
    lo = xyz.amin(0) - pad
    hi = xyz.amax(0) + pad
    voxel = ((hi - lo).max() / res).clamp_min(1e-8)
    dx, dy, dz = (torch.ceil((hi - lo) / voxel).long() + 1).tolist()

    sinv = _inverse_covariance(scale, quat)
    kreq = torch.ceil(3.0 * scale.amax(-1) / voxel).long().clamp(1, int(kernel))
    sharp = color_sharpen != 1.0
    vol = torch.zeros(dx * dy * dz, device=device)
    colvol = torch.zeros(dx * dy * dz, 3, device=device, dtype=col_dtype)
    wcol = torch.zeros(dx * dy * dz, device=device, dtype=col_dtype) if sharp else None
    for k in range(1, int(kernel) + 1):
        sel = (kreq == k).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        rng = torch.arange(-k, k + 1, device=device, dtype=torch.float32)
        off = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
        for st in range(0, sel.numel(), chunk):
            gi = sel[st:st + chunk]
            cc = xyz[gi]
            idx = ((cc - lo) / voxel).round()[:, None, :] + off[None]
            d = (lo + idx * voxel) - cc[:, None, :]
            quad = torch.einsum("bmi,bij,bmj->bm", d, sinv[gi], d)
            wgt = opacity[gi, None] * torch.exp(-0.5 * quad)
            wgt = torch.where(quad < 9.0, wgt, torch.zeros_like(wgt))  # clip beyond 3 sigma
            ii = idx.long()
            ix = ii[..., 0].clamp(0, dx - 1)
            iy = ii[..., 1].clamp(0, dy - 1)
            iz = ii[..., 2].clamp(0, dz - 1)
            flat = (ix * (dy * dz) + iy * dz + iz).reshape(-1)
            vol.index_add_(0, flat, wgt.reshape(-1))
            wp = wgt.pow(color_sharpen) if sharp else wgt
            colvol.index_add_(0, flat, (wp[..., None] * rgb[gi, None, :]).reshape(-1, 3).to(col_dtype))
            if sharp:
                wcol.index_add_(0, flat, wp.reshape(-1).to(col_dtype))
    colnorm = (wcol if sharp else vol).reshape(dx, dy, dz)
    return (vol.reshape(dx, dy, dz), colvol.reshape(dx, dy, dz, 3), colnorm,
            lo.detach().cpu().numpy(), float(voxel))


def _otsu_level(values, bins=256):
    """Otsu threshold: density value that best splits inside/outside."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5
    w = np.cumsum(hist)
    mu = np.cumsum(hist * centers)
    wf = w[-1] - w
    mb = mu / np.where(w > 0, w, 1.0)
    mf = (mu[-1] - mu) / np.where(wf > 0, wf, 1.0)
    var_b = w * wf * (mb - mf) ** 2
    var_b[(w <= 0) | (wf <= 0)] = -1.0
    return float(centers[int(np.argmax(var_b))])


def _surface_nets(vol, level, voxel, origin, device):
    """Vectorized Surface Nets: one dual vertex per sign-changing cell at its
    edge-crossing mean, quads wound outward. Returns (verts world, faces)."""
    vol = vol.to(device=device, dtype=torch.float32)
    dx, dy, dz = vol.shape
    origin_t = torch.as_tensor(origin, device=device, dtype=torch.float32)
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64))
    if dx < 2 or dy < 2 or dz < 2:
        return empty

    inside = vol >= level
    cs8 = [inside[ox:ox + dx - 1, oy:oy + dy - 1, oz:oz + dz - 1]
           for ox, oy, oz in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                              (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))]
    any_in = cs8[0] | cs8[1] | cs8[2] | cs8[3] | cs8[4] | cs8[5] | cs8[6] | cs8[7]
    all_in = cs8[0] & cs8[1] & cs8[2] & cs8[3] & cs8[4] & cs8[5] & cs8[6] & cs8[7]
    active = any_in & ~all_in
    nv = int(active.sum())
    if nv == 0:
        return empty

    del any_in, all_in, cs8
    ac = active.nonzero(as_tuple=False)
    offs = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                         [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], device=device)
    offf = offs.to(torch.float32)
    edges = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                          [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]], device=device)
    e0, e1 = edges[:, 0], edges[:, 1]
    oe0, oe1 = offf[e0], offf[e1]

    cstep = 1 << 18
    loc = []
    for st in range(0, nv, cstep):
        ci = ac[st:st + cstep, None, :] + offs[None]
        cval = vol[ci[..., 0], ci[..., 1], ci[..., 2]]
        csl = cval >= level
        v0, v1 = cval[:, e0], cval[:, e1]
        cross = (csl[:, e0] != csl[:, e1])[..., None].to(torch.float32)
        denom = v1 - v0
        t = torch.where(denom.abs() > 1e-12, (level - v0) / denom, torch.full_like(denom, 0.5)).clamp(0, 1)
        pts = torch.lerp(oe0, oe1, t[..., None])
        loc.append((pts * cross).sum(1) / cross.sum(1).clamp_min(1.0))
    local = torch.cat(loc, 0) if len(loc) > 1 else loc[0]
    verts = origin_t + (ac.to(torch.float32) + local) * voxel
    del loc, local

    vid = torch.full((dx - 1, dy - 1, dz - 1), -1, dtype=torch.int32, device=device)
    vid[active] = torch.arange(nv, dtype=torch.int32, device=device)
    del active

    faces = []

    def emit(cr, sol, a, b, d, c):
        valid = cr & (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        if not bool(valid.any()):
            return
        a, b, c, d, sol = a[valid], b[valid], c[valid], d[valid], sol[valid]
        p2, p4 = torch.where(sol, b, c), torch.where(sol, c, b)
        faces.append(torch.stack([a, p2, d], 1))
        faces.append(torch.stack([a, d, p4], 1))

    a = inside[0:dx - 1, 1:dy - 1, 1:dz - 1]
    emit(a != inside[1:dx, 1:dy - 1, 1:dz - 1], a,
         vid[:, 0:dy - 2, 0:dz - 2], vid[:, 1:dy - 1, 0:dz - 2],
         vid[:, 1:dy - 1, 1:dz - 1], vid[:, 0:dy - 2, 1:dz - 1])
    a = inside[1:dx - 1, 0:dy - 1, 1:dz - 1]
    emit(a != inside[1:dx - 1, 1:dy, 1:dz - 1], a,
         vid[0:dx - 2, :, 0:dz - 2], vid[0:dx - 2, :, 1:dz - 1],
         vid[1:dx - 1, :, 1:dz - 1], vid[1:dx - 1, :, 0:dz - 2])
    a = inside[1:dx - 1, 1:dy - 1, 0:dz - 1]
    emit(a != inside[1:dx - 1, 1:dy - 1, 1:dz], a,
         vid[0:dx - 2, 0:dy - 2, :], vid[1:dx - 1, 0:dy - 2, :],
         vid[1:dx - 1, 1:dy - 1, :], vid[0:dx - 2, 1:dy - 1, :])

    if not faces:
        return empty
    return verts.cpu().numpy().astype(np.float32), torch.cat(faces, 0).cpu().numpy().astype(np.int64)


def _taubin_smooth(verts, faces, iters, lam=0.5, mu=-0.53):
    """Volume-preserving lambda|mu smoothing (no Laplacian shrinkage)."""
    if iters <= 0 or len(verts) == 0 or len(faces) == 0:
        return verts
    nv = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)
    adj = coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    adj.data[:] = 1.0
    deg = np.clip(np.asarray(adj.sum(1)).ravel(), 1.0, None).astype(np.float32)[:, None]
    v = verts.astype(np.float32)
    for _ in range(int(iters)):
        for fac in (lam, mu):
            v = v + np.float32(fac) * ((adj @ v) / deg - v)
    return np.ascontiguousarray(v)


def _clean_components(verts, faces, min_verts):
    """Drop floaters (< min_verts) and the inner shell of the double wall."""
    nv = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    ncomp, label = connected_components(
        coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)), directed=False)
    flabel = label[faces[:, 0]]
    keep = np.bincount(label, minlength=ncomp) >= min_verts
    if keep.sum() > 1:
        fcount = np.bincount(flabel, minlength=ncomp)
        largest = np.where(keep, fcount, -1).argmax()
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        cvol = np.bincount(flabel, weights=np.einsum("ij,ij->i", v0, np.cross(v1, v2)), minlength=ncomp)
        cidx = np.arange(ncomp)
        cmin = np.stack([_ndi_min(verts[:, a], label, cidx) for a in range(3)], 1)
        cmax = np.stack([_ndi_max(verts[:, a], label, cidx) for a in range(3)], 1)
        tol = 1e-4 * (cmax[largest] - cmin[largest]).max()
        enclosed = (cmin >= cmin[largest] - tol).all(1) & (cmax <= cmax[largest] + tol).all(1)
        inner = enclosed & (np.sign(cvol) != np.sign(cvol[largest])) & (np.arange(ncomp) != largest)
        keep &= ~inner
    faces = faces[keep[flabel]]
    if len(faces) == 0:
        return verts[:0], faces
    used = np.unique(faces)
    remap = np.full(nv, -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def splat_to_mesh(xyz, opacity, scale, quat, rgb, *, resolution, device,
                  kernel=5, level_bias=0.4, min_component=500, min_opacity=0.02,
                  color_sharpen=2.0, taubin=12, scale_floor_frac=0.9, vivid_color=False):
    """Full splat -> (verts f32, faces i64, colors f32 0..1, display space). Returns
    None if no surface. All inputs are torch tensors on `device`: xyz (N,3), opacity
    (N,), scale (N,3) linear std, quat (N,4) wxyz, rgb (N,3) display 0..1."""
    opacity = opacity.reshape(-1)
    keep = opacity >= min_opacity
    xyz, scale, quat, opacity, rgb = xyz[keep], scale[keep], quat[keep], opacity[keep], rgb[keep]
    if xyz.shape[0] == 0:
        return None

    # Floor each Gaussian's density footprint to a fraction of the inter-Gaussian
    # spacing. TripoSplat's surfels are often sub-voxel, so without this the density
    # grid fragments into thousands of disconnected spikes (the "holes"). Spacing is
    # a property of the cloud (not the grid), so this stays solid at any resolution.
    # Geometry/density only — colour weighting is unaffected.
    if scale_floor_frac > 0 and xyz.shape[0] > 8:
        xn = xyz.detach().cpu().numpy()
        sub = xn[np.random.default_rng(0).choice(len(xn), min(20000, len(xn)), replace=False)]
        spacing = float(np.median(cKDTree(xn).query(sub, k=2)[0][:, 1]))
        scale = torch.clamp(scale, min=spacing * scale_floor_frac)

    vol, colvol, colnorm, origin, voxel = _splat_density(
        xyz, opacity, scale, quat, rgb, resolution, kernel, device, color_sharpen=color_sharpen)
    colvol_np = colvol.float().cpu().numpy()
    colnorm_np = colnorm.float().cpu().numpy()
    del colvol, colnorm

    vmin, vmax = float(vol.min()), float(vol.max())
    occ = vol[vol > vmax * 1e-3]
    if occ.numel() == 0:
        return None
    level = min(max(_otsu_level(occ.cpu().numpy()) * level_bias, vmin + 1e-6 * (vmax - vmin)),
                vmax - 1e-6 * (vmax - vmin))

    verts, faces = _surface_nets(vol, level, voxel, origin, device)
    del vol
    if min_component > 0 and len(faces) > 0:
        verts, faces = _clean_components(verts, faces, min_component)
    if len(verts) == 0 or len(faces) == 0:
        return None

    verts = _taubin_smooth(verts, faces, taubin)

    # Trilinearly sample the (smooth) colour volume — this is what keeps the surface
    # colour clean instead of speckled.
    coords = ((verts - origin) / voxel).T
    num = np.stack([map_coordinates(colvol_np[..., c], coords, order=1, mode="nearest") for c in range(3)], -1)
    den = map_coordinates(colnorm_np, coords, order=1, mode="nearest")
    col = np.clip(num / np.clip(den, 1e-8, None)[:, None], 0.0, 1.0).astype(np.float32)

    # Vivid: luminance-scaled saturation pre-boost to offset the viewer's tone-mapping
    # desaturation. Darks left neutral so dark features (eyes) don't pick up a cast.
    if vivid_color:
        luma = (col @ np.array([0.299, 0.587, 0.114], np.float32))[:, None]
        amt = np.float32(0.5) * np.clip(luma / np.float32(0.4), 0.0, 1.0)
        col = np.clip(luma + (col - luma) * (1.0 + amt), 0.0, 1.0).astype(np.float32)

    # NO sRGB->linear: Modly's viewer displays vertex colours as-is (it does not
    # re-encode linear->sRGB), so a sRGB->linear pass here only darkens everything
    # (measured ~0.24 -> ~0.10 mean brightness). Raw display colours match the viewer.
    return verts.astype(np.float32), faces.astype(np.int64), col

"""
TSDF fusion of splat expected-depth maps — "mesh the appearance, not the density".

The density iso-surface fails wherever the gaussian field is locally weak (the
source-image blind spots): craters, shreds, crevasses. But the splat's RENDERED
depth is reliable from every direction — so treat the splat as K virtual RGB-D
cameras and run classic volumetric TSDF fusion (KinectFusion-style):

  for each voxel x, each view i seeing it:
      sdf_i(x) = splat_depth_i(project_i(x)) - z_cam_i(x)     (+ outside, - inside)
      fuse truncated sdf with per-view weights

  surface = zero level set (Surface Nets — reused as-is)

Per-view weights drop: background rays carve (visual hull), depth-discontinuity
pixels are masked (expected depth is wrong across jumps), behind-surface samples
are only trusted within the truncation band. Voxels no view can see (object
interior) fall back to the density volume sign so the mesh stays closed.

Colours: same co-splatted colour volume as the density path (trilinear sampling).
"""
import numpy as np
import torch
import torch.nn.functional as F


def _depth_views(gaussians, points_ref, n_views, res, device):
    """Render splat depth/alpha/reliability for a full-coverage rig."""
    from .render.cameras import make_rig, cam_to_torch
    from .render.splat import render_splats

    GOLD = float(np.pi * (3.0 - np.sqrt(5.0)))
    cams = make_rig(points_ref, k=n_views, res=res, phase=GOLD / 5, z_max=0.96)
    views = []
    for cam in cams:
        camt = cam_to_torch(cam, device)
        _, a, d = render_splats(gaussians["xyz"], gaussians["opacity"].reshape(-1),
                                gaussians["scale"], gaussians["quat"],
                                gaussians["rgb"], camt, return_depth=True)
        # reliability: depth is meaningless across discontinuities (3x3 range test)
        dd = d[None, None]
        rng = (F.max_pool2d(dd, 3, 1, 1) + F.max_pool2d(-dd, 3, 1, 1))[0, 0]
        fg = a > 0.5
        med = rng[fg].median() if bool(fg.any()) else rng.median()
        reliable = rng < 3.0 * med.clamp_min(1e-6)
        views.append((camt, a, d, reliable))
    return views


@torch.no_grad()
def tsdf_fuse(gaussians, vol, origin, voxel, device,
              n_views: int = 24, view_res: int = 512,
              trunc_voxels: float = 3.0, level: float = None,
              chunk: int = 2_000_000, vox_mask: torch.Tensor = None,
              views=None):
    """Fuse splat depth maps into a TSDF over the SAME grid as `vol`.

    gaussians : torch dict, scales already floored (the renderable field)
    vol       : (X,Y,Z) density grid — interior fallback for never-seen voxels
    level     : density level for the interior fallback sign
    Returns tsdf (X,Y,Z) float32, NEGATIVE inside (Surface Nets convention:
    extract `inside = tsdf <= 0` via level 0 on -tsdf).
    """
    X, Y, Z = vol.shape
    trunc = trunc_voxels * voxel

    if views is None:
        centers = gaussians["xyz"].cpu().numpy()
        views = _depth_views(gaussians, centers, n_views, view_res, device)

    ax = torch.arange(X, device=device, dtype=torch.float32)
    ay = torch.arange(Y, device=device, dtype=torch.float32)
    az = torch.arange(Z, device=device, dtype=torch.float32)
    org = torch.as_tensor(origin, device=device, dtype=torch.float32)

    tsdf = torch.zeros(X * Y * Z, device=device)
    wsum = torch.zeros(X * Y * Z, device=device)

    gx, gy, gz = torch.meshgrid(ax, ay, az, indexing="ij")
    pts_all = (torch.stack([gx, gy, gz], -1).reshape(-1, 3) * voxel) + org
    del gx, gy, gz

    # restrict fusion to the masked voxels (defect-driven repair: cost scales
    # with the defect size, not the grid)
    if vox_mask is not None:
        sel_idx = vox_mask.reshape(-1).nonzero(as_tuple=True)[0]
        pts_all = pts_all[sel_idx]
    else:
        sel_idx = None

    for camt, alpha, depth, reliable in views:
        Hv = Wv = int(camt["res"])
        amap = alpha[None, None]
        dmap = depth[None, None]
        rmap = reliable.float()[None, None]
        for s in range(0, len(pts_all), chunk):
            p = pts_all[s:s + chunk]
            pc = p @ camt["R"].T + camt["t"]
            z = pc[:, 2]
            front = z > 1e-4
            u = camt["fx"] * pc[:, 0] / z.clamp_min(1e-6) + camt["cx"]
            v = camt["fy"] * pc[:, 1] / z.clamp_min(1e-6) + camt["cy"]
            # normalized grid coords for bilinear sampling
            gn = torch.stack([u / (Wv - 1) * 2 - 1, v / (Hv - 1) * 2 - 1], -1)
            gn = gn.reshape(1, 1, -1, 2)
            a_s = F.grid_sample(amap, gn, align_corners=True)[0, 0, 0]
            d_s = F.grid_sample(dmap, gn, align_corners=True)[0, 0, 0]
            r_s = F.grid_sample(rmap, gn, align_corners=True)[0, 0, 0]
            inb = front & (u >= 0) & (u <= Wv - 1) & (v >= 0) & (v <= Hv - 1)

            sdf = d_s - z                                      # + in front of surface
            # background ray (alpha low): everything along it is OUTSIDE
            bg = inb & (a_s < 0.3)
            # surface ray, reliable depth: trusted within the truncation band;
            # far behind the surface the view knows nothing -> weight 0
            fgm = inb & (a_s > 0.7) & (r_s > 0.5) & (sdf > -trunc)

            val = (sdf / trunc).clamp(-1.0, 1.0)
            w = torch.zeros_like(val)
            w[bg] = 0.5                                        # carving, soft
            w[fgm] = 1.0
            idx = torch.arange(s, s + len(p), device=device)
            if sel_idx is not None:
                idx = sel_idx[idx]
            tsdf.scatter_add_(0, idx, w * torch.where(bg, torch.ones_like(val), val))
            wsum.scatter_add_(0, idx, w)

    seen = wsum > 1e-6
    out = torch.where(seen, tsdf / wsum.clamp_min(1e-6),
                      torch.zeros_like(tsdf))
    # interior fallback for never-seen voxels: density sign keeps the shell closed
    if level is not None:
        inside = (vol.reshape(-1) >= level) & ~seen
        outside = ~seen & ~inside
        out[inside] = -1.0
        out[outside] = 1.0
    return out.reshape(X, Y, Z)


@torch.no_grad()
def splat_to_mesh_repair(xyz, opacity, scale, quat, rgb, *, resolution, device,
                         taubin: int = 6, n_views: int = 24, view_res: int = 512,
                         min_component: int = 500, min_opacity: float = 0.02,
                         scale_floor_frac: float = 0.9, vol_smooth: float = 0.7,
                         defect_voxels: float = 3.0, dilate_voxels: int = 5):
    """Defect-driven extraction repair — surgical version of the hybrid.

    1. Standard density extraction (identical to splat_to_mesh incl. vol_smooth).
    2. Defect detection: render the mesh depth vs the splat expected depth on the
       fusion rig; pixels where the mesh sits > defect_voxels BEHIND the splat
       (reliable, both covered) are sunken pockets (source-image blind spots).
       If none: return the standard mesh untouched (bit-identical, ~free).
    3. Back-project defect pixels at splat depth, mark + dilate those voxels,
       fuse TSDF ONLY there, blend, re-extract.

    Returns (verts f32, faces i64, colors f32, n_defect_px int).
    """
    import sys as _sys
    from pathlib import Path as _P
    _ext = _P(__file__).parent.parent
    if str(_ext) not in _sys.path:
        _sys.path.insert(0, str(_ext))
    from splat_mesh import (_splat_density, _otsu_level, _surface_nets,
                            _clean_components, _taubin_smooth, _gaussian_blur3d,
                            _filament_stamp)
    from scipy.spatial import cKDTree
    from .render.mesh import render_mesh

    opacity = opacity.reshape(-1)
    keep = opacity >= min_opacity
    xyz, scale, quat, opacity, rgb = xyz[keep], scale[keep], quat[keep], opacity[keep], rgb[keep]
    if xyz.shape[0] == 0:
        return None
    xn = xyz.detach().cpu().numpy()
    sub = xn[np.random.default_rng(0).choice(len(xn), min(20000, len(xn)), replace=False)]
    spacing = float(np.median(cKDTree(xn).query(sub, k=2)[0][:, 1]))
    scale = torch.clamp(scale, min=spacing * scale_floor_frac)
    g = {"xyz": xyz, "opacity": opacity, "scale": scale, "quat": quat, "rgb": rgb}

    vol, colvol, colnorm, origin, voxel = _splat_density(
        xyz, opacity, scale, quat, rgb, resolution, 5, device)
    vol_s = _gaussian_blur3d(vol, vol_smooth) if vol_smooth > 0 else vol
    colnorm_s = _gaussian_blur3d(colnorm.float(), vol_smooth) if vol_smooth > 0 else colnorm
    colvol_s = (torch.stack([_gaussian_blur3d(colvol[..., ch].float(), vol_smooth)
                             for ch in range(3)], dim=-1) if vol_smooth > 0 else colvol)
    occ = vol_s[vol_s > float(vol_s.max()) * 1e-3]
    if occ.numel() == 0:
        return None
    level = _otsu_level(occ.cpu().numpy()) * 0.4

    # filament (whisker) recovery — same pass as splat_to_mesh, no-op when clean
    vol_s, _nfil = _filament_stamp(vol_s, level, xyz, origin, voxel, device)

    v0, f0 = _surface_nets(vol_s, level, voxel, origin, device)
    if min_component > 0 and len(f0) > 0:
        v0, f0 = _clean_components(v0, f0, min_component)
    if len(v0) == 0 or len(f0) == 0:
        return None

    # ── defect detection on the fusion rig ────────────────────────────────
    views = _depth_views(g, xn, n_views, view_res, device)
    vt = torch.as_tensor(v0, device=device)
    ft = torch.as_tensor(f0, device=device)
    dummy = torch.zeros(len(vt), 3, device=device)
    defect_pts = []
    for camt, a, d, reliable in views:
        _, m_a, aux = render_mesh(vt, ft, dummy, camt, return_aux=True)
        a_f, d_f, r_f = a.reshape(-1), d.reshape(-1), reliable.reshape(-1)
        # two defect kinds: sunken pockets (mesh well behind the splat) and
        # true holes (splat surface visible where the mesh has nothing at all —
        # the shredded-extraction failure mode)
        sunk = (aux["hit"] & (a_f > 0.7) & r_f
                & (aux["depth"] - d_f > defect_voxels * voxel))
        # true holes must be THICK regions: erode 2px to kill the 1-2px
        # silhouette slivers every mesh/splat edge mismatch produces (false
        # positives that made the repair fire on clean objects)
        H = W = int(camt["res"])
        hole_img = ((~aux["hit"]) & (a_f > 0.7) & r_f).float().reshape(1, 1, H, W)
        hole = (-F.max_pool2d(-hole_img, 5, stride=1, padding=2))[0, 0].reshape(-1) > 0.5
        bad = sunk | hole
        if not bool(bad.any()):
            continue
        H = W = int(camt["res"])
        ii = bad.nonzero(as_tuple=True)[0]
        u = (ii % W).float(); vpx = (ii // W).float()
        z = d_f[ii]
        pc = torch.stack([(u - camt["cx"]) / camt["fx"] * z,
                          (vpx - camt["cy"]) / camt["fy"] * z, z], 1)
        defect_pts.append((pc - camt["t"]) @ camt["R"])     # world (R orthonormal)
    n_defect = sum(len(p) for p in defect_pts)
    print(f"[repair] defect pixels: {n_defect}", flush=True)

    if n_defect < 200:                                       # nothing to repair
        v0 = _taubin_smooth(v0, f0, taubin)
        col = _sample_colors_dilated(colvol_s, colnorm_s, v0, origin, voxel)
        return v0.astype(np.float32), f0.astype(np.int64), col, n_defect

    # ── mark + dilate defect voxels, fuse TSDF only there ────────────────
    X, Y, Z = vol_s.shape
    org = torch.as_tensor(origin, device=device, dtype=torch.float32)
    pts = torch.cat(defect_pts)
    ijk = ((pts - org) / voxel).round().long()
    ok = ((ijk >= 0) & (ijk < torch.tensor([X, Y, Z], device=device))).all(1)
    ijk = ijk[ok]
    mask = torch.zeros(X, Y, Z, device=device)
    mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1.0
    k = 2 * int(dilate_voxels) + 1
    mask = F.max_pool3d(mask[None, None], k, stride=1, padding=k // 2)[0, 0] > 0

    tsdf = tsdf_fuse(g, vol_s, origin, voxel, device, level=level,
                     vox_mask=mask, views=views)
    F_d = ((level - vol_s) / max(level, 1e-8)).clamp(-1.0, 1.0)
    field = torch.where(mask, tsdf, F_d)
    del tsdf, F_d

    verts, faces = _surface_nets(-field, 0.0, voxel, origin, device)
    del field
    if min_component > 0 and len(faces) > 0:
        verts, faces = _clean_components(verts, faces, min_component)
    if len(verts) == 0 or len(faces) == 0:
        return None
    verts = _taubin_smooth(verts, faces, taubin)
    col = _sample_colors_dilated(colvol_s, colnorm_s, verts, origin, voxel)
    return verts.astype(np.float32), faces.astype(np.int64), col, n_defect


@torch.no_grad()
def splat_to_mesh_hybrid(xyz, opacity, scale, quat, rgb, *, resolution, device,
                         taubin: int = 6, n_views: int = 24, view_res: int = 512,
                         min_component: int = 500, min_opacity: float = 0.02,
                         scale_floor_frac: float = 0.9, trust_tau: float = 1.0):
    """Confidence-blended extraction: density field where the density is locally
    trustworthy, TSDF (fused splat depth) where it is not.

    Rationale (gated separately): the density iso-surface wins almost everywhere
    (sharp edges, thin parts, colour alignment) but collapses in source-image
    blind spots (craters); raw TSDF is immune to craters but loses edges/colour
    fidelity globally. Both are signed fields on the SAME grid, so blend per
    voxel by local density evidence:

        F_d(x)  = clamp((level - vol)/level, -1, 1)       density signed field
        w(x)    = clamp(maxpool_blur(vol)/(tau*level), 0, 1)   local trust
        F(x)    = w*F_d + (1-w)*tsdf       -> Surface Nets at level 0 on -F

    In a crater the whole neighbourhood sits below the level -> w<1 -> the TSDF
    restores the perceived surface; where density is solid w=1 and the result is
    bit-identical to the density extraction.
    """
    import sys as _sys
    from pathlib import Path as _P
    _ext = _P(__file__).parent.parent
    if str(_ext) not in _sys.path:
        _sys.path.insert(0, str(_ext))
    from splat_mesh import (_splat_density, _otsu_level, _surface_nets,
                            _clean_components, _taubin_smooth, _gaussian_blur3d)
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree

    opacity = opacity.reshape(-1)
    keep = opacity >= min_opacity
    xyz, scale, quat, opacity, rgb = xyz[keep], scale[keep], quat[keep], opacity[keep], rgb[keep]
    if xyz.shape[0] == 0:
        return None
    xn = xyz.detach().cpu().numpy()
    sub = xn[np.random.default_rng(0).choice(len(xn), min(20000, len(xn)), replace=False)]
    spacing = float(np.median(cKDTree(xn).query(sub, k=2)[0][:, 1]))
    scale = torch.clamp(scale, min=spacing * scale_floor_frac)
    g = {"xyz": xyz, "opacity": opacity, "scale": scale, "quat": quat, "rgb": rgb}

    vol, colvol, colnorm, origin, voxel = _splat_density(
        xyz, opacity, scale, quat, rgb, resolution, 5, device)
    vol_s = _gaussian_blur3d(vol, 0.7)
    occ = vol_s[vol_s > float(vol_s.max()) * 1e-3]
    if occ.numel() == 0:
        return None
    level = _otsu_level(occ.cpu().numpy()) * 0.4

    F_d = ((level - vol_s) / max(level, 1e-8)).clamp(-1.0, 1.0)
    # local trust: does the neighbourhood reach the level anywhere?
    ev = F.max_pool3d(vol_s[None, None], 7, stride=1, padding=3)[0, 0]
    w = (ev / (trust_tau * level)).clamp(0.0, 1.0)
    frac_blend = float((w < 0.99).float().mean())
    print(f"[hybrid] blended voxels: {frac_blend * 100:.1f}%", flush=True)

    tsdf = tsdf_fuse(g, vol_s, origin, voxel, device,
                     n_views=n_views, view_res=view_res, level=level)
    field = w * F_d + (1.0 - w) * tsdf
    del vol, vol_s, F_d, tsdf, w, ev

    verts, faces = _surface_nets(-field, 0.0, voxel, origin, device)
    del field
    if min_component > 0 and len(faces) > 0:
        verts, faces = _clean_components(verts, faces, min_component)
    if len(verts) == 0 or len(faces) == 0:
        return None
    verts = _taubin_smooth(verts, faces, taubin)

    col = _sample_colors_dilated(colvol, colnorm, verts, origin, voxel)
    return verts.astype(np.float32), faces.astype(np.int64), col


def _sample_colors_dilated(colvol, colnorm, verts, origin, voxel):
    """Trilinear colour sampling with nearest-valid dilation.

    Surfaces restored in source-image blind spots sit where the co-splatted
    colour volume is empty (colnorm ~ 0) — raw num/den sampling there produces
    concentric ring artifacts. Dilate first: every invalid voxel takes the
    colour of its nearest valid voxel (single EDT)."""
    from scipy.ndimage import map_coordinates, distance_transform_edt

    cv = colvol.float().cpu().numpy()
    cn = colnorm.float().cpu().numpy()
    valid = cn > max(1e-6, 1e-4 * float(cn.max()))
    if not valid.all() and valid.any():
        ind = distance_transform_edt(~valid, return_distances=False,
                                     return_indices=True)
        cv = cv[ind[0], ind[1], ind[2]]
        cn = cn[ind[0], ind[1], ind[2]]
    coords = ((verts - origin) / voxel).T
    num = np.stack([map_coordinates(cv[..., ch], coords, order=1, mode="nearest")
                    for ch in range(3)], -1)
    den = map_coordinates(cn, coords, order=1, mode="nearest")
    return np.clip(num / np.clip(den, 1e-8, None)[:, None], 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def splat_to_mesh_tsdf(xyz, opacity, scale, quat, rgb, *, resolution, device,
                       taubin: int = 6, n_views: int = 24, view_res: int = 512,
                       min_component: int = 500, min_opacity: float = 0.02,
                       scale_floor_frac: float = 0.9):
    """Full splat -> mesh via TSDF fusion of the splat's own depth renders.

    Same signature spirit / colour handling as splat_mesh.splat_to_mesh; the
    difference is WHAT gets meshed: the perceived surface (fused expected-depth)
    instead of the density iso-surface. Immune by construction to local density
    weakness (craters in source-image blind spots, shreds, crevasses).
    Returns (verts f32, faces i64, colors f32 0..1) or None.
    """
    import sys as _sys
    from pathlib import Path as _P
    _ext = _P(__file__).parent.parent
    if str(_ext) not in _sys.path:
        _sys.path.insert(0, str(_ext))
    from splat_mesh import (_splat_density, _otsu_level, _surface_nets,
                            _clean_components, _taubin_smooth, _gaussian_blur3d)
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree

    opacity = opacity.reshape(-1)
    keep = opacity >= min_opacity
    xyz, scale, quat, opacity, rgb = xyz[keep], scale[keep], quat[keep], opacity[keep], rgb[keep]
    if xyz.shape[0] == 0:
        return None
    xn = xyz.detach().cpu().numpy()
    sub = xn[np.random.default_rng(0).choice(len(xn), min(20000, len(xn)), replace=False)]
    spacing = float(np.median(cKDTree(xn).query(sub, k=2)[0][:, 1]))
    scale = torch.clamp(scale, min=spacing * scale_floor_frac)

    g = {"xyz": xyz, "opacity": opacity, "scale": scale, "quat": quat, "rgb": rgb}
    vol, colvol, colnorm, origin, voxel = _splat_density(
        xyz, opacity, scale, quat, rgb, resolution, 5, device)
    vol_s = _gaussian_blur3d(vol, 0.7)
    occ = vol_s[vol_s > float(vol_s.max()) * 1e-3]
    if occ.numel() == 0:
        return None
    level = _otsu_level(occ.cpu().numpy()) * 0.4

    tsdf = tsdf_fuse(g, vol_s, origin, voxel, device,
                     n_views=n_views, view_res=view_res, level=level)
    del vol, vol_s

    verts, faces = _surface_nets(-tsdf, 0.0, voxel, origin, device)
    del tsdf
    if min_component > 0 and len(faces) > 0:
        verts, faces = _clean_components(verts, faces, min_component)
    if len(verts) == 0 or len(faces) == 0:
        return None
    verts = _taubin_smooth(verts, faces, taubin)

    cv = colvol.float().cpu().numpy()
    cn = colnorm.float().cpu().numpy()
    coords = ((verts - origin) / voxel).T
    num = np.stack([map_coordinates(cv[..., ch], coords, order=1, mode="nearest")
                    for ch in range(3)], -1)
    den = map_coordinates(cn, coords, order=1, mode="nearest")
    col = np.clip(num / np.clip(den, 1e-8, None)[:, None], 0.0, 1.0).astype(np.float32)
    return verts.astype(np.float32), faces.astype(np.int64), col

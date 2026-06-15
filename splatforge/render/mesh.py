"""
Pure-torch z-buffer mesh rasterizer with interpolated vertex colors (eval-only),
plus per-pixel UV texture sampling from the aux buffers.

Unlit on purpose: the splat reference composite is also unlit DC colour, so the
two images are directly comparable. Z-test winner per pixel is resolved with the
pack-trick: quantized depth in the high 32 bits of an int64, fragment id in the
low 32 — a single scatter_reduce(amin) gives the front-most fragment per pixel.

Not differentiable. The differentiable version (soft rasterizer or nvdiffrast)
is Phase B-1 work, only if the B-0 gate shows recoverable signal.
"""
import numpy as np
import torch


@torch.no_grad()
def sample_texture(aux: dict, uvs: torch.Tensor, tex: torch.Tensor) -> torch.Tensor:
    """Per-pixel bilinear texture lookup from render_mesh aux buffers.
    uvs (V,2) in [0,1]; tex (T,T,3) float in [0,1]. Returns (H*W,3), 0 off-mesh.
    glTF convention: v up, image row down (matches texture_bake's row flip)."""
    uv = (aux["bary"][:, 0:1] * uvs[aux["vid"][:, 0]]
          + aux["bary"][:, 1:2] * uvs[aux["vid"][:, 1]]
          + aux["bary"][:, 2:3] * uvs[aux["vid"][:, 2]]).clamp(0, 1)
    T = tex.shape[0]
    x = uv[:, 0] * (T - 1)
    y = (1.0 - uv[:, 1]) * (T - 1)
    x0 = x.floor().long().clamp(0, T - 2)
    y0 = y.floor().long().clamp(0, T - 2)
    fx = (x - x0.float()).unsqueeze(1)
    fy = (y - y0.float()).unsqueeze(1)
    rgb = (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x0 + 1] * fx * (1 - fy)
           + tex[y0 + 1, x0] * (1 - fx) * fy + tex[y0 + 1, x0 + 1] * fx * fy)
    rgb[~aux["hit"]] = 0.0
    return rgb


@torch.no_grad()
def render_mesh(verts, faces, colors, cam: dict, face_chunk: int = 8192,
                return_aux: bool = False):
    """Rasterize a vertex-colored mesh from one camera.

    verts (V,3) float32, faces (F,3) int64, colors (V,3) float32 in [0,1] —
    torch tensors on the same device. cam as in cameras.make_rig (torch R/t).
    Returns (rgb (H,W,3), alpha (H,W)) float32. Background = 0.

    return_aux: additionally return {vid (H*W,3) int64 vertex ids of the winning
    face, bary (H*W,3) float32, hit (H*W,) bool} — pixel colour is exactly
    sum_j bary_j * colors[vid_j] on hit pixels, i.e. LINEAR in vertex colors.
    Used by the render-supervised colour solver (Phase B-1).
    """
    dev = verts.device
    H = W = int(cam["res"])
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

    p_cam = verts @ cam["R"].T + cam["t"]                  # (V,3)
    z = p_cam[:, 2].clamp_min(1e-6)
    sx = fx * p_cam[:, 0] / z + cx
    sy = fy * p_cam[:, 1] / z + cy
    scr = torch.stack([sx, sy], 1)                          # (V,2)

    # depth packing: 32-bit quantized z (front = small) | 32-bit fragment payload
    zmin, zmax = float(z.min()), float(z.max())
    zspan = max(zmax - zmin, 1e-6)

    best = torch.full((H * W,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=dev)
    # payload buffers grow per chunk; store face id + barycentrics of the winning frag
    frag_face = []
    frag_bary = []
    frag_z = []
    frag_pix = []
    frag_id0 = 0

    F = len(faces)
    for s in range(0, F, face_chunk):
        e = min(s + face_chunk, F)
        f = faces[s:e]
        v0, v1, v2 = scr[f[:, 0]], scr[f[:, 1]], scr[f[:, 2]]      # (B,2)
        z0, z1, z2 = z[f[:, 0]], z[f[:, 1]], z[f[:, 2]]

        # backface + signed area (screen y is down, keep both winds: no culling —
        # Surface Nets output is consistently wound but play safe for eval)
        area = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) \
             - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        ok = area.abs() > 1e-9

        lo = torch.minimum(torch.minimum(v0, v1), v2).floor().clamp(min=0)
        hi = torch.maximum(torch.maximum(v0, v1), v2).ceil()
        hi[:, 0] = hi[:, 0].clamp(max=W - 1)
        hi[:, 1] = hi[:, 1].clamp(max=H - 1)
        wbox = (hi[:, 0] - lo[:, 0] + 1).clamp_min(0)
        hbox = (hi[:, 1] - lo[:, 1] + 1).clamp_min(0)
        npix = torch.where(ok, (wbox * hbox).long(), torch.zeros_like(wbox).long())
        total = int(npix.sum())
        if total == 0:
            continue

        # fragment -> (face-in-chunk, pixel-in-bbox)
        fid = torch.repeat_interleave(torch.arange(e - s, device=dev), npix)
        cum = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), npix.cumsum(0)[:-1]])
        loc = torch.arange(total, device=dev) - cum[fid]
        bw = wbox[fid].long()
        pxx = lo[fid, 0].long() + (loc % bw)
        pyy = lo[fid, 1].long() + (loc // bw)

        # barycentric at pixel center
        pc = torch.stack([pxx.float() + 0.5, pyy.float() + 0.5], 1)
        a0, a1, a2 = v0[fid], v1[fid], v2[fid]
        den = area[fid]
        w0 = ((a1[:, 0] - pc[:, 0]) * (a2[:, 1] - pc[:, 1])
              - (a1[:, 1] - pc[:, 1]) * (a2[:, 0] - pc[:, 0])) / den
        w1 = ((a2[:, 0] - pc[:, 0]) * (a0[:, 1] - pc[:, 1])
              - (a2[:, 1] - pc[:, 1]) * (a0[:, 0] - pc[:, 0])) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not bool(inside.any()):
            continue

        fid, pxx, pyy = fid[inside], pxx[inside], pyy[inside]
        w0, w1, w2 = w0[inside], w1[inside], w2[inside]

        # perspective-correct depth: interpolate 1/z
        izf = (w0 / z0[fid] + w1 / z1[fid] + w2 / z2[fid]).clamp_min(1e-9)
        zf = 1.0 / izf
        zq = ((zf - zmin) / zspan * (2**31 - 2)).long().clamp(0, 2**31 - 2)

        gid = torch.arange(len(fid), device=dev) + frag_id0
        packed = (zq << 32) | gid
        pix = pyy * W + pxx
        best.scatter_reduce_(0, pix, packed, reduce="amin")

        frag_face.append(faces[s:e][fid])
        frag_bary.append(torch.stack([w0, w1, w2], 1))
        frag_z.append(zf)
        frag_pix.append(pix)
        frag_id0 += len(fid)

    rgb = torch.zeros(H * W, 3, device=dev)
    alpha = torch.zeros(H * W, device=dev)
    if frag_id0 == 0:
        if return_aux:
            aux = {"vid": torch.zeros(H * W, 3, dtype=torch.int64, device=dev),
                   "bary": torch.zeros(H * W, 3, device=dev),
                   "depth": torch.zeros(H * W, device=dev),
                   "hit": torch.zeros(H * W, dtype=torch.bool, device=dev)}
            return rgb.reshape(H, W, 3), alpha.reshape(H, W), aux
        return rgb.reshape(H, W, 3), alpha.reshape(H, W)

    all_face = torch.cat(frag_face)
    all_bary = torch.cat(frag_bary)

    hit = best != torch.iinfo(torch.int64).max
    win = (best[hit] & 0xFFFFFFFF).long()                  # winning fragment id

    fwin = all_face[win]                                    # (P,3) vertex ids
    bwin = all_bary[win]                                    # (P,3)
    col = (bwin[:, 0:1] * colors[fwin[:, 0]]
           + bwin[:, 1:2] * colors[fwin[:, 1]]
           + bwin[:, 2:3] * colors[fwin[:, 2]])
    rgb[hit] = col.clamp(0, 1)
    alpha[hit] = 1.0
    if return_aux:
        all_z = torch.cat(frag_z)
        vid = torch.zeros(H * W, 3, dtype=torch.int64, device=dev)
        bar = torch.zeros(H * W, 3, device=dev)
        dep = torch.zeros(H * W, device=dev)
        vid[hit] = fwin
        bar[hit] = bwin
        dep[hit] = all_z[win]
        return rgb.reshape(H, W, 3), alpha.reshape(H, W), \
               {"vid": vid, "bary": bar, "depth": dep, "hit": hit}
    return rgb.reshape(H, W, 3), alpha.reshape(H, W)

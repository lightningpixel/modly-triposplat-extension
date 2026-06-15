"""
Phase B-1 candidate: render-supervised vertex colours.

Key property: with geometry frozen, the rasterized pixel colour is LINEAR in the
vertex colours (fixed barycentric weights). Fitting vertex colours to the splat
reference renders is therefore a sparse linear least-squares problem — no
differentiable rasterizer needed.

    min_c  sum_px || B c - ref ||^2  +  lam_lap |L c|^2  +  lam_anchor |c - c0|^2

  - B: per-pixel barycentric gather (3 vertices per observed pixel)
  - L: uniform mesh-graph Laplacian — propagates colour into vertices never seen
    by any training view and keeps the field smooth
  - anchor: weak pull toward the initial colours, so unobserved regions keep a
    sane base instead of drifting

Solved with Adam on the accumulated pixel observations (fits VRAM easily:
~1-2M observations x 3 weights). Training views MUST be disjoint from the
held-out eval rig (use cameras.make_rig(phase=...)).
"""
import numpy as np
import torch

from .splat import render_splats
from .mesh import render_mesh
from .cameras import cam_to_torch


def _edge_list(faces: torch.Tensor):
    """Directed edge list (6F,) src/dst for the uniform Laplacian."""
    src = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2],
                     faces[:, 1], faces[:, 2], faces[:, 0]])
    dst = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0],
                     faces[:, 0], faces[:, 1], faces[:, 2]])
    return src, dst


def collect_observations(verts, faces, colors0, gaussians, train_cams, device):
    """Render splat refs + rasterize the mesh on each training view; gather the
    per-pixel (vertex ids, barycentric weights, reference rgb) observations.

    Only pixels where BOTH renders have coverage supervise colour: splat-only
    pixels are geometry misses (a colour cannot fix them), mesh-only pixels have
    no reference signal.
    Returns (vid (P,3) int64, bw (P,3) float32, ref (P,3) float32).
    """
    vids, bws, refs = [], [], []
    for cam in train_cams:
        camt = cam_to_torch(cam, device)
        s_rgb, s_a = render_splats(gaussians["xyz"], gaussians["opacity"].reshape(-1),
                                   gaussians["scale"], gaussians["quat"],
                                   gaussians["rgb"], camt)
        _, m_a, aux = render_mesh(verts, faces, colors0, camt, return_aux=True)
        sup = aux["hit"] & (s_a.reshape(-1) > 0.5)
        if not bool(sup.any()):
            continue
        vids.append(aux["vid"][sup])
        bws.append(aux["bary"][sup])
        refs.append(s_rgb.reshape(-1, 3)[sup])
    assert vids, "no supervised pixels — check camera rig / geometry"
    return torch.cat(vids), torch.cat(bws), torch.cat(refs)


def refine_colors(verts, faces, colors0, gaussians, train_cams,
                  iters: int = 400, lr: float = 0.02,
                  lam_lap: float = 0.10, lam_anchor: float = 0.02) -> torch.Tensor:
    """Fit vertex colours to the splat renders. Returns (V,3) float32 in [0,1].

    All tensor args on the same device. colors0 is both the init and the anchor.
    """
    device = verts.device
    vid, bw, ref = collect_observations(verts, faces, colors0, gaussians,
                                        train_cams, device)
    n_obs = len(ref)
    seen = torch.zeros(len(verts), dtype=torch.bool, device=device)
    seen[vid.reshape(-1)] = True
    print(f"[color_opt] {n_obs} pixel observations  "
          f"({int(seen.sum())}/{len(verts)} vertices observed)", flush=True)

    src, dst = _edge_list(faces)
    deg = torch.zeros(len(verts), 1, device=device)
    deg.scatter_add_(0, src.unsqueeze(1), torch.ones(len(src), 1, device=device))
    deg = deg.clamp_min(1.0)

    c = colors0.clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)

    for it in range(iters):
        opt.zero_grad()
        pred = (bw[:, 0:1] * c[vid[:, 0]]
                + bw[:, 1:2] * c[vid[:, 1]]
                + bw[:, 2:3] * c[vid[:, 2]])
        loss_data = ((pred - ref) ** 2).mean()

        nbr = torch.zeros_like(c)
        nbr.scatter_add_(0, src.unsqueeze(1).expand(-1, 3), c[dst])
        loss_lap = ((c - nbr / deg) ** 2).mean()

        loss_anchor = ((c - colors0) ** 2).mean()

        loss = loss_data + lam_lap * loss_lap + lam_anchor * loss_anchor
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = c.clamp(0.0, 1.0).detach()
    print(f"[color_opt] done: data_loss {float(loss_data.detach()):.5f}", flush=True)
    return out


def refine_texture(verts, faces, uvs, tex0, gaussians, train_cams,
                   iters: int = 300, lr: float = 0.02,
                   lam_anchor: float = 0.004, lam_tv: float = 0.04) -> torch.Tensor:
    """Fit TEXTURE texels to the splat renders. Returns (T,T,3) float32 in [0,1].

    Same linearity trick as refine_colors, one level down: a rendered pixel is the
    bilinear blend of 4 texels at its interpolated UV — fixed weights once geometry
    and UVs are frozen. Corrects the baked texture's tone bias (KNN colour estimator
    vs the splat composite) while keeping its sharpness.

    verts/faces: unwrapped mesh (vmapping applied). uvs (V,2) in [0,1].
    tex0 (T,T,3) float32 in [0,1] — init + anchor (keeps unobserved texels sane).
    """
    device = verts.device
    T = tex0.shape[0]

    idx_list, w_list, ref_list = [], [], []
    for cam in train_cams:
        camt = cam_to_torch(cam, device)
        s_rgb, s_a = render_splats(gaussians["xyz"], gaussians["opacity"].reshape(-1),
                                   gaussians["scale"], gaussians["quat"],
                                   gaussians["rgb"], camt)
        dummy = torch.zeros(len(verts), 3, device=device)
        _, m_a, aux = render_mesh(verts, faces, dummy, camt, return_aux=True)
        sup = aux["hit"] & (s_a.reshape(-1) > 0.5)
        if not bool(sup.any()):
            continue
        uv = (aux["bary"][sup, 0:1] * uvs[aux["vid"][sup, 0]]
              + aux["bary"][sup, 1:2] * uvs[aux["vid"][sup, 1]]
              + aux["bary"][sup, 2:3] * uvs[aux["vid"][sup, 2]]).clamp(0, 1)
        x = uv[:, 0] * (T - 1)
        y = (1.0 - uv[:, 1]) * (T - 1)            # glTF v-up -> image row-down
        x0 = x.floor().long().clamp(0, T - 2)
        y0 = y.floor().long().clamp(0, T - 2)
        fx, fy = x - x0.float(), y - y0.float()
        idx_list.append(torch.stack([y0 * T + x0, y0 * T + x0 + 1,
                                     (y0 + 1) * T + x0, (y0 + 1) * T + x0 + 1], 1))
        w_list.append(torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy),
                                   (1 - fx) * fy, fx * fy], 1))
        ref_list.append(s_rgb.reshape(-1, 3)[sup])

    assert idx_list, "no supervised pixels — check camera rig / geometry"
    idx = torch.cat(idx_list)                     # (P,4)
    w   = torch.cat(w_list)                       # (P,4)
    ref = torch.cat(ref_list)                     # (P,3)
    print(f"[tex_opt] {len(ref)} pixel observations on a {T}x{T} texture", flush=True)

    t0_flat = tex0.reshape(-1, 3)
    t = t0_flat.clone().requires_grad_(True)
    opt = torch.optim.Adam([t], lr=lr)

    for _ in range(iters):
        opt.zero_grad()
        pred = (w[:, 0:1] * t[idx[:, 0]] + w[:, 1:2] * t[idx[:, 1]]
                + w[:, 2:3] * t[idx[:, 2]] + w[:, 3:4] * t[idx[:, 3]])
        loss_data = ((pred - ref) ** 2).mean()
        loss_anchor = ((t - t0_flat) ** 2).mean()
        timg = t.reshape(T, T, 3)
        loss_tv = (((timg[1:] - timg[:-1]) ** 2).mean()
                   + ((timg[:, 1:] - timg[:, :-1]) ** 2).mean())
        loss = loss_data + lam_anchor * loss_anchor + lam_tv * loss_tv
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = t.reshape(T, T, 3).clamp(0.0, 1.0).detach()
    print(f"[tex_opt] done: data_loss {float(loss_data.detach()):.5f}", flush=True)
    return out


def refine_texture_production(verts_np: np.ndarray, faces_np: np.ndarray,
                              uvs_np: np.ndarray, tex_uint8: np.ndarray,
                              gaussians: dict, device,
                              n_train_views: int = 20, res: int = 1024) -> np.ndarray:
    """Production entry point for baked-texture refinement (uint8 in / uint8 out).

    gaussians: torch dict {xyz, opacity, scale, quat, rgb}, opacity>=0.02 filtered.
    Scales floored to 0.9 x median spacing for the splat reference renders.

    Validated config (2026-06-11): train res 1024, 20 views z_max 0.95, anchor
    0.004, tv 0.04 — texrefine BEATS the vertex-colour standard on voiture
    (+0.18 dB / +0.0045 SSIM) and ties PSNR on the cat with the occluded-region
    speckle patches strongly reduced. Lower train res / higher anchor leaves
    bake noise visible in regions the source image never saw (backs of heads).
    """
    from scipy.spatial import cKDTree
    from .cameras import make_rig

    g = {k: v.detach().float().to(device) for k, v in gaussians.items()}
    g["opacity"] = g["opacity"].reshape(-1)
    xn = g["xyz"].cpu().numpy()
    sub = xn[np.random.default_rng(0).choice(len(xn), min(20000, len(xn)),
                                             replace=False)]
    spacing = float(np.median(cKDTree(xn).query(sub, k=2)[0][:, 1]))
    g["scale"] = g["scale"].clamp_min(0.9 * spacing)

    GOLDEN = float(np.pi * (3.0 - np.sqrt(5.0)))
    train_cams = make_rig(np.asarray(verts_np, np.float32), k=n_train_views,
                          res=res, phase=GOLDEN / 2, z_max=0.95)

    verts = torch.as_tensor(np.asarray(verts_np, np.float32), device=device)
    faces = torch.as_tensor(np.asarray(faces_np, np.int64), device=device)
    uvs = torch.as_tensor(np.asarray(uvs_np, np.float32), device=device)
    tex0 = torch.as_tensor(tex_uint8.astype(np.float32) / 255.0, device=device)

    out = refine_texture(verts, faces, uvs, tex0, g, train_cams)
    return (out.cpu().numpy() * 255.0).round().astype(np.uint8)


def refine_colors_production(verts_np: np.ndarray, faces_np: np.ndarray,
                             colors_np: np.ndarray, gaussians: dict,
                             device, n_train_views: int = 20,
                             res: int = 512) -> np.ndarray:
    """Production entry point (numpy in / numpy out, splat/Z-up frame).

    gaussians: dict of torch tensors {xyz, opacity, scale, quat, rgb}, already
    filtered to opacity >= 0.02 (same as the reconstruction). Scales are floored
    to 0.9 x median spacing here so the splat reference renders the same field
    the mesh was extracted from.

    Phase B gate (2026-06-11, held-out views): +1.01 / +0.46 / +0.08 dB PSNR and
    SSIM up on all three test objects vs production trilinear colours.
    """
    from scipy.spatial import cKDTree
    from .cameras import make_rig

    g = {k: v.detach().float().to(device) for k, v in gaussians.items()}
    g["opacity"] = g["opacity"].reshape(-1)

    xn = g["xyz"].cpu().numpy()
    sub = xn[np.random.default_rng(0).choice(len(xn), min(20000, len(xn)),
                                             replace=False)]
    spacing = float(np.median(cKDTree(xn).query(sub, k=2)[0][:, 1]))
    g["scale"] = g["scale"].clamp_min(0.9 * spacing)

    GOLDEN = float(np.pi * (3.0 - np.sqrt(5.0)))
    train_cams = make_rig(np.asarray(verts_np, np.float32), k=n_train_views,
                          res=res, phase=GOLDEN / 2, z_max=0.95)

    verts = torch.as_tensor(np.asarray(verts_np, np.float32), device=device)
    faces = torch.as_tensor(np.asarray(faces_np, np.int64), device=device)
    colors0 = torch.as_tensor(np.clip(np.asarray(colors_np, np.float32), 0, 1),
                              device=device)

    out = refine_colors(verts, faces, colors0, g, train_cams)
    return out.cpu().numpy().astype(np.float32)

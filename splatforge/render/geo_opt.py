"""
DepthFit — depth-supervised linearized mesh refinement.

Goal: make the mesh SURFACE match the splat's perceived surface (the alpha-
weighted expected depth of the gaussian composite, which is SMOOTH), killing
the lumpy/scarred relief that survives volume extraction and ruins the lit
appearance, without a differentiable rasterizer.

Linearity trick (third member of the family, after vertex colours and texels):
with the rasterization frozen (winning face + barycentrics fixed per pixel) and
each vertex constrained to move along its own normal, v'_j = v_j + s_j n_j, the
mesh depth at a supervised pixel is LINEAR in the scalar field s:

    depth(p) = sum_j b_j (R(v_j + s_j n_j) + t)_z
             = mesh_depth(p) + sum_j [ b_j (R n_j)_z ] s_j

so  min_s  sum_p w_p (depth(p) - splat_depth(p))^2
         + lam_lap |L s|^2 + lam_anchor |s|^2

is a sparse linear least-squares solved with Adam. Visibility changes as the
surface moves, so we re-rasterize and recompute normals for a few outer rounds.

Robustness:
  - pixels where |mesh_depth - splat_depth| > trim x median are occlusion/
    silhouette mismatches a normal displacement cannot fix -> dropped
  - grazing pixels are down-weighted by |n . view|
  - s is clamped per round (default 1.5 x mean edge length)
  - normal-only motion prevents tangential sliding / fold-overs; the Laplacian
    on the SCALAR field propagates into unseen regions smoothly
"""
import numpy as np
import torch

from .cameras import cam_to_torch
from .splat import render_splats
from .mesh import render_mesh


def _vertex_normals(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    vn = torch.zeros_like(v)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = torch.linalg.cross(p1 - p0, p2 - p0)
    for k in range(3):
        vn.scatter_add_(0, f[:, k:k + 1].expand(-1, 3), fn)
    return torch.nn.functional.normalize(vn, dim=1)


def _edge_list(f: torch.Tensor):
    src = torch.cat([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    dst = torch.cat([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    return src, dst


@torch.no_grad()
def _splat_depths(gaussians, train_cams, device, disc_k: float = 3.0):
    """Reference depth maps (rendered once — gaussians are frozen).

    Also returns a per-pixel reliability mask: the alpha-weighted EXPECTED depth
    is wrong near depth discontinuities (it averages across the jump — at a box
    edge or past a thin leg the target lands mid-air and the fit rounds/distorts
    the geometry; measured −2 dB on the wooden car without this mask). A pixel
    is reliable iff the local 3x3 depth range is below disc_k x its own scale.
    """
    import torch.nn.functional as F
    out = []
    for cam in train_cams:
        camt = cam_to_torch(cam, device)
        _, s_a, s_d = render_splats(
            gaussians["xyz"], gaussians["opacity"].reshape(-1), gaussians["scale"],
            gaussians["quat"], gaussians["rgb"], camt, return_depth=True)
        H = W = int(camt["res"])
        d = s_d[None, None]                                   # (1,1,H,W)
        dmax = F.max_pool2d(d, 3, stride=1, padding=1)
        dmin = -F.max_pool2d(-d, 3, stride=1, padding=1)
        rng = (dmax - dmin)[0, 0]
        fg = s_a > 0.5
        med = rng[fg].median() if bool(fg.any()) else rng.median()
        reliable = (rng < disc_k * med.clamp_min(1e-6)).reshape(-1)
        out.append((camt, s_a.reshape(-1), s_d.reshape(-1), reliable))
    return out


def refine_geometry(verts, faces, gaussians, train_cams,
                    rounds: int = 3, iters: int = 200, lr: float = None,
                    lam_lap: float = 2.0, lam_anchor: float = 0.05,
                    trim: float = 4.0) -> torch.Tensor:
    """Fit the mesh surface to the splat expected-depth maps.

    verts (V,3) f32 torch, faces (F,3) i64 torch, gaussians: torch dict with
    scales ALREADY floored to 0.9 x spacing (the field the splat renders).
    Returns refined verts (V,3) torch.
    """
    device = verts.device
    v = verts.clone()
    src, dst = _edge_list(faces)
    deg = torch.zeros(len(v), 1, device=device)
    deg.scatter_add_(0, src.unsqueeze(1), torch.ones(len(src), 1, device=device))
    deg = deg.clamp_min(1.0)

    mel = float((v[faces[:, 1]] - v[faces[:, 0]]).norm(dim=1).mean())
    s_clamp = 1.5 * mel
    if lr is None:
        lr = 0.1 * mel

    refs = _splat_depths(gaussians, train_cams, device)

    for rnd in range(rounds):
        n = _vertex_normals(v, faces)

        vid_l, w_l, r0_l, px_w = [], [], [], []
        for camt, s_a, s_d, reliable in refs:
            dummy = torch.zeros(len(v), 3, device=device)
            _, m_a, aux = render_mesh(v, faces, dummy, camt, return_aux=True)
            sup = aux["hit"] & (s_a > 0.5) & reliable
            if not bool(sup.any()):
                continue
            vid = aux["vid"][sup]                       # (P,3)
            bw = aux["bary"][sup]                       # (P,3)
            resid0 = aux["depth"][sup] - s_d[sup]       # current depth error

            # robust trim: occlusion mismatches can't be fixed by displacement
            med = resid0.abs().median()
            keep = resid0.abs() < trim * med.clamp_min(1e-5)

            # coefficient of s_j on this pixel: b_j * (R n_j)_z
            nz = (n @ camt["R"].T)[:, 2]                # (V,)
            coef = bw * nz[vid]                         # (P,3)
            # grazing pixels: normal nearly orthogonal to view -> weak, noisy
            grazing_w = (bw * nz[vid].abs()).sum(1).clamp(0, 1)

            vid_l.append(vid[keep])
            w_l.append(coef[keep])
            r0_l.append(resid0[keep])
            px_w.append(grazing_w[keep])

        assert vid_l, "no supervised pixels"
        vid = torch.cat(vid_l); w = torch.cat(w_l)
        r0 = torch.cat(r0_l);  pw = torch.cat(px_w)
        print(f"[geo_opt] round {rnd + 1}/{rounds}: {len(r0)} px obs  "
              f"depth_mae={float(r0.abs().mean()):.5f}", flush=True)

        s = torch.zeros(len(v), 1, device=device, requires_grad=True)
        opt = torch.optim.Adam([s], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            ds = (w * s[vid].squeeze(-1)).sum(1)        # (P,) linear in s
            loss_data = (pw * (r0 + ds) ** 2).mean()

            nbr = torch.zeros_like(s)
            nbr.scatter_add_(0, src.unsqueeze(1), s[dst])
            loss_lap = ((s - nbr / deg) ** 2).mean()

            loss = loss_data + lam_lap * loss_lap + lam_anchor * (s ** 2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            step = s.clamp(-s_clamp, s_clamp)
            v = v + step * n
            print(f"[geo_opt]   |s| mean={float(step.abs().mean()):.5f} "
                  f"max={float(step.abs().max()):.5f}", flush=True)

    return v

"""
Auto colour-path selection: an embedded mini-gate.

No geometric heuristic separates the cases (measured: UV-seam duplication ratio
does NOT predict which path wins), so the only reliable selector is empirical —
render BOTH refined candidates on a small held-out validation rig and keep the
winner. Same principle as the Phase B bench, shrunk to production cost.

Validation rig: 4 views, azimuth phase distinct from BOTH the refine train rig
(GOLDEN/2) and the bench eval rig (0), elevation trimmed like eval. Renders are
2x supersampled then average-pooled: real viewers mipmap their textures, and a
point-sampled comparison would punish the texture candidate with aliasing noise
(measured: -0.02 SSIM artifact, enough to flip the verdict).

Decision: higher mean SSIM wins (the metric that catches seam/aliasing failures);
PSNR is the tie-breaker.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .cameras import make_rig, cam_to_torch
from .splat import render_splats
from .mesh import render_mesh, sample_texture
from .metrics import view_metrics

_GOLDEN = float(np.pi * (3.0 - np.sqrt(5.0)))


def _pool(img, k):
    if img.dim() == 2:
        return F.avg_pool2d(img[None, None], k)[0, 0]
    return F.avg_pool2d(img.permute(2, 0, 1)[None], k)[0].permute(1, 2, 0)


@torch.no_grad()
def _eval_candidate(verts, faces, colors, tex_pack, gaussians, cams, ssaa, device):
    psnr, ssim = [], []
    for cam in cams:
        cam = dict(cam)
        for k_ in ("fx", "fy", "cx", "cy"):
            cam[k_] = cam[k_] * ssaa
        cam["res"] = int(cam["res"] * ssaa)
        camt = cam_to_torch(cam, device)
        s_rgb, s_a = render_splats(gaussians["xyz"], gaussians["opacity"].reshape(-1),
                                   gaussians["scale"], gaussians["quat"],
                                   gaussians["rgb"], camt)
        m_rgb, m_a, aux = render_mesh(verts, faces, colors, camt, return_aux=True)
        if tex_pack is not None:
            res_px = int(camt["res"])
            m_rgb = sample_texture(aux, tex_pack[0], tex_pack[1]).reshape(res_px, res_px, 3)
        if ssaa > 1:
            s_rgb, s_a = _pool(s_rgb, ssaa), _pool(s_a, ssaa)
            m_rgb, m_a = _pool(m_rgb, ssaa), _pool(m_a, ssaa)
        m = view_metrics(s_rgb, s_a, m_rgb, m_a)
        psnr.append(m["psnr"]); ssim.append(m["ssim"])
    return float(np.mean(psnr)), float(np.mean(ssim))


@torch.no_grad()
def _depth_metrics(verts, faces, gaussians, cams, device):
    """Mean depth-RMSE + IoU of a mesh vs the splat on the given cams."""
    rmse, iou = [], []
    for cam in cams:
        camt = cam_to_torch(cam, device)
        _, s_a, s_d = render_splats(gaussians["xyz"], gaussians["opacity"].reshape(-1),
                                    gaussians["scale"], gaussians["quat"],
                                    gaussians["rgb"], camt, return_depth=True)
        dummy = torch.zeros(len(verts), 3, device=device)
        _, m_a, aux = render_mesh(verts, faces, dummy, camt, return_aux=True)
        sm = s_a.reshape(-1) > 0.5
        mm = aux["hit"]
        inter = sm & mm
        if bool(inter.any()):
            d = (s_d.reshape(-1) - aux["depth"])[inter]
            rmse.append(float(torch.sqrt((d ** 2).mean())))
        iou.append(float(inter.sum()) / max(float((sm | mm).sum()), 1.0))
    return float(np.mean(rmse)) if rmse else float("inf"), float(np.mean(iou))


@torch.no_grad()
def guard_geometry(verts_before, verts_after, faces, gaussians, device,
                   n_val_views: int = 4, res: int = 512) -> bool:
    """Per-object acceptance test for a geometry refinement (e.g. DepthFit).

    DepthFit helps organic/damaged surfaces dramatically (+0.2..+3.1 dB) but
    HURTS hard-edged objects (−2 dB on the wooden car: the splat expected depth
    is unreliable at edges and the per-view targets conflict). No static rule
    separates them — so, as with the colour paths, each object validates the
    refinement on fresh held-out views and keeps it only if depth fidelity
    improves without losing silhouette.
    """
    cams = make_rig(verts_before.cpu().numpy(), k=n_val_views, res=res,
                    phase=_GOLDEN / 4)
    r0, i0 = _depth_metrics(verts_before, faces, gaussians, cams, device)
    r1, i1 = _depth_metrics(verts_after, faces, gaussians, cams, device)
    ok = (r1 < r0 * 0.97) and (i1 > i0 - 0.005)
    print(f"[geo guard] depth_rmse {r0:.5f} -> {r1:.5f}  iou {i0:.3f} -> {i1:.3f}"
          f"  => {'ACCEPT' if ok else 'REJECT'}", flush=True)
    return ok


@torch.no_grad()
def pick_color_path(vertex_mesh: tuple, texture_mesh: tuple,
                    gaussians: dict, device,
                    n_val_views: int = 4, res: int = 512, ssaa: int = 2) -> dict:
    """Pick the better colour path on a fresh validation rig.

    vertex_mesh : (verts (V,3) f32 np, faces (F,3) np, colors (V,3) f32 np 0..1)
    texture_mesh: (verts, faces, uvs (V,2) f32 np, tex_uint8 (T,T,3) np)
    gaussians   : torch dict, opacity-filtered AND 0.9x-spacing-floored scales
                  (the same field the splat reference must render).

    Returns {"winner": "vertex"|"texture", "vertex": (psnr, ssim),
             "texture": (psnr, ssim)}.
    """
    vv = torch.as_tensor(np.asarray(vertex_mesh[0], np.float32), device=device)
    vf = torch.as_tensor(np.asarray(vertex_mesh[1], np.int64), device=device)
    vc = torch.as_tensor(np.clip(np.asarray(vertex_mesh[2], np.float32), 0, 1),
                         device=device)

    tv = torch.as_tensor(np.asarray(texture_mesh[0], np.float32), device=device)
    tf = torch.as_tensor(np.asarray(texture_mesh[1], np.int64), device=device)
    uvs = torch.as_tensor(np.asarray(texture_mesh[2], np.float32), device=device)
    tex = torch.as_tensor(texture_mesh[3].astype(np.float32) / 255.0, device=device)
    dummy = torch.zeros(len(tv), 3, device=device)

    # Multi-scale: the viewer shows the object SMALL (mip 2-3), where a
    # fragmented UV atlas bleeds across island borders. A full-frame 512 eval
    # overestimates the texture path (measured: texture won the gate on a cat
    # whose shattered atlas looked dirty in the viewer; at 128 px the same
    # texture lost its lead). Score both candidates at full and quarter res and
    # decide on the MEAN SSIM — vertex colours are scale-stable, a healthy
    # atlas keeps its win, a shattered one is caught.
    v_ps, v_ss, t_ps, t_ss = [], [], [], []
    for r in (res, max(res // 4, 96)):
        cams = make_rig(np.asarray(vertex_mesh[0], np.float32), k=n_val_views,
                        res=r, phase=_GOLDEN / 3)
        vp, vs = _eval_candidate(vv, vf, vc, None, gaussians, cams, ssaa, device)
        tp, ts = _eval_candidate(tv, tf, dummy, (uvs, tex), gaussians, cams,
                                 ssaa, device)
        v_ps.append(vp); v_ss.append(vs); t_ps.append(tp); t_ss.append(ts)
        print(f"[auto] res={r}: vertex psnr={vp:.2f} ssim={vs:.4f} | "
              f"texture psnr={tp:.2f} ssim={ts:.4f}", flush=True)
    v_psnr, v_ssim = float(np.mean(v_ps)), float(np.mean(v_ss))
    t_psnr, t_ssim = float(np.mean(t_ps)), float(np.mean(t_ss))

    if abs(t_ssim - v_ssim) > 1e-4:
        winner = "texture" if t_ssim > v_ssim else "vertex"
    else:
        winner = "texture" if t_psnr >= v_psnr else "vertex"
    print(f"[auto] mean: vertex ssim={v_ssim:.4f} | texture ssim={t_ssim:.4f}"
          f" -> {winner}", flush=True)
    return {"winner": winner, "vertex": (v_psnr, v_ssim),
            "texture": (t_psnr, t_ssim)}

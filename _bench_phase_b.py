"""
Phase B bench — render-based evaluation: mesh vs splat reference on held-out views.

Usage:
  python _bench_phase_b.py [--npz PATH] [--obj LABEL] [--views K] [--res N] [--dump]

Gate criterion (Phase B, when a candidate refinement exists):
  candidate mean PSNR/SSIM > Standard mean PSNR/SSIM over the SAME K held-out views.
  This run establishes the Standard baseline (and sanity-checks the render stack).

Input: self-contained .gaussians.npz (start_verts/start_faces required; start_colors
used if present, else colors fall back to GaussianField.color KNN sampling — flagged
in output since production colors use trilinear volume sampling instead).

--dump writes per-view PNG pairs to splatforge/render/bench_out/ for visual review.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch

EXT       = Path(__file__).parent
INSTALLED = Path(r"C:\Users\LORCHIE\Documents\Modly\extensions\triposplat")
sys.path.insert(0, str(INSTALLED))
sys.path.insert(0, str(EXT))

parser = argparse.ArgumentParser()
parser.add_argument("--npz",   default="", help="Self-contained .gaussians.npz")
parser.add_argument("--obj",   default="obj")
parser.add_argument("--views", type=int, default=8)
parser.add_argument("--res",   type=int, default=512)
parser.add_argument("--dump",  action="store_true", help="Write per-view PNGs")
parser.add_argument("--candidate", default="",
                    choices=["", "colors", "smooth", "volsmooth", "texture", "texrefine", "geo", "tsdf"],
                    help="Phase B candidate to gate against Standard. volsmooth "
                         "re-extracts BOTH sides at full production res (224); "
                         "texture bakes a 2K UV texture (production baked path).")
parser.add_argument("--train-views", type=int, default=16,
                    help="Training views for the candidate (azimuth-offset rig)")
parser.add_argument("--sigma", type=float, default=1.0,
                    help="volsmooth: density-volume blur sigma in voxels")
parser.add_argument("--sigma-color", type=float, default=None,
                    help="volsmooth: colour-volume blur sigma (default: same as --sigma)")
parser.add_argument("--train-res", type=int, default=None,
                    help="texrefine: training render resolution (default: --res)")
parser.add_argument("--tex-anchor", type=float, default=0.02)
parser.add_argument("--tex-tv", type=float, default=0.01)
parser.add_argument("--bake-smooth", action="store_true",
                    help="texrefine: bake the init texture with 0.9x-floored scales "
                         "(smooth tone-true init; refine adds back sharpness)")
parser.add_argument("--subsample", type=int, default=1,
                    help="Keep every Nth gaussian — simulates low-count decodes "
                         "(4 ~= 65k from a 262k npz) for robustness tests")
parser.add_argument("--ssaa", type=int, default=1,
                    help="Supersample factor for eval renders (2 = render at 2x res, "
                         "average-pool). Removes the no-mipmap aliasing bias against "
                         "textures; real viewers mipmap.")
args = parser.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert args.npz, "--npz PATH required"
npz_path = Path(args.npz)
assert npz_path.exists(), f"npz not found: {npz_path}"
data = np.load(str(npz_path))

assert "start_verts" in data.files, (
    f"{npz_path.name}: no embedded start mesh — re-generate with the updated extension.")

_GAUSS_KEYS = {"xyz", "opacity", "scale", "quat", "rgb"}
g = {k: torch.as_tensor(data[k], dtype=torch.float32).to(dev)
     for k in data.files if k in _GAUSS_KEYS}
_keep = g["opacity"].reshape(-1) >= 0.02          # same filter as splat_to_mesh
g = {k: v[_keep] for k, v in g.items()}
if args.subsample > 1:
    g = {k: v[::args.subsample] for k, v in g.items()}
print(f"Gaussians: {len(g['xyz'])} (opacity>=0.02"
      + (f", subsampled 1/{args.subsample}" if args.subsample > 1 else "") + ")",
      flush=True)

verts_np = np.ascontiguousarray(data["start_verts"], np.float32)
faces_np = np.ascontiguousarray(data["start_faces"], np.int64)
verts = torch.as_tensor(verts_np, device=dev)
faces = torch.as_tensor(faces_np, device=dev)
print(f"Mesh: {len(faces_np)} faces", flush=True)

# ── mesh colors: embedded if available, else KNN field sampling (flagged) ────
from splatforge.fields import GaussianField
if "start_colors" in data.files:
    _c = data["start_colors"]
    _c = _c.astype(np.float32) / 255.0 if _c.dtype == np.uint8 else _c.astype(np.float32)
    colors = torch.as_tensor(_c, device=dev)
    color_src = "embedded (production trilinear)"
else:
    field = GaussianField(g, dev, surface_pts=verts_np)
    colors = field.color(verts).clamp(0, 1)
    color_src = "KNN-field (FALLBACK — production uses trilinear volume sampling; "\
                "absolute PSNR has a colour bias, comparisons between candidates "\
                "sharing this colouring remain fair)"
print(f"Colors: {color_src}", flush=True)

# ── splat scale floor: render the same field the mesh was extracted from ─────
# (sub-voxel surfels render as dust; the reconstruction floored them too)
from scipy.spatial import cKDTree
scale_raw = g["scale"].clone()          # un-floored — splat_to_mesh floors itself
_xn = g["xyz"].cpu().numpy()
_sub = _xn[np.random.default_rng(0).choice(len(_xn), min(20000, len(_xn)), replace=False)]
_spacing = float(np.median(cKDTree(_xn).query(_sub, k=2)[0][:, 1]))
g["scale"] = g["scale"].clamp_min(0.9 * _spacing)

# ── render rig + eval loop ────────────────────────────────────────────────────
from splatforge.render import make_rig, cam_to_torch, render_splats, render_mesh, view_metrics

cams = make_rig(verts_np, k=args.views, res=args.res)

OUT = EXT / "splatforge" / "render" / "bench_out"
if args.dump:
    OUT.mkdir(parents=True, exist_ok=True)


_splat_cache = {}   # view idx -> (rgb, alpha, depth); identical across candidates


def _shaded(eval_verts, eval_faces, aux, camt):
    """Headlight-shaded mesh render (H,W) — makes crevasse-type geometry noise
    visible in the dumps (the colour render is unlit and nearly blind to it)."""
    vn = torch.zeros_like(eval_verts)
    p0, p1, p2 = eval_verts[eval_faces[:, 0]], eval_verts[eval_faces[:, 1]], eval_verts[eval_faces[:, 2]]
    fn = torch.linalg.cross(p1 - p0, p2 - p0)
    for k in range(3):
        vn.scatter_add_(0, eval_faces[:, k:k+1].expand(-1, 3), fn)
    vn = torch.nn.functional.normalize(vn, dim=1) @ camt["R"].T
    nz = (aux["bary"][:, 0:1] * vn[aux["vid"][:, 0]]
          + aux["bary"][:, 1:2] * vn[aux["vid"][:, 1]]
          + aux["bary"][:, 2:3] * vn[aux["vid"][:, 2]])[:, 2].abs()
    img = torch.full((aux["hit"].shape[0],), 0.5, device=eval_verts.device)
    img[aux["hit"]] = (0.15 + 0.85 * nz[aux["hit"]]).clamp(0, 1)
    res = int(camt["res"])
    return img.reshape(res, res)


def _tex_sample(aux, uvs_t, tex_t):
    """Per-pixel bilinear texture lookup from the rasterizer aux buffers.
    glTF convention: v up, image row down (matches texture_bake's row flip)."""
    uv = (aux["bary"][:, 0:1] * uvs_t[aux["vid"][:, 0]]
          + aux["bary"][:, 1:2] * uvs_t[aux["vid"][:, 1]]
          + aux["bary"][:, 2:3] * uvs_t[aux["vid"][:, 2]]).clamp(0, 1)
    T = tex_t.shape[0]
    x = uv[:, 0] * (T - 1)
    y = (1.0 - uv[:, 1]) * (T - 1)
    x0 = x.floor().long().clamp(0, T - 2)
    y0 = y.floor().long().clamp(0, T - 2)
    fx = (x - x0.float()).unsqueeze(1)
    fy = (y - y0.float()).unsqueeze(1)
    rgb = (tex_t[y0, x0] * (1 - fx) * (1 - fy) + tex_t[y0, x0 + 1] * fx * (1 - fy)
           + tex_t[y0 + 1, x0] * (1 - fx) * fy + tex_t[y0 + 1, x0 + 1] * fx * fy)
    rgb[~aux["hit"]] = 0.0
    return rgb


def _pool(img, k):
    """Average-pool (H,W) or (H,W,3) by factor k."""
    import torch.nn.functional as F
    if img.dim() == 2:
        return F.avg_pool2d(img[None, None], k)[0, 0]
    return F.avg_pool2d(img.permute(2, 0, 1)[None], k)[0].permute(1, 2, 0)


def evaluate(tag: str, eval_verts: torch.Tensor, eval_faces: torch.Tensor,
             eval_colors: torch.Tensor, eval_tex=None) -> dict:
    """Render splat ref (cached) + mesh candidate on the held-out rig.
    eval_tex: optional (uvs (V,2), texture (T,T,3)) — overrides vertex colours.
    With --ssaa N, renders at N x res and average-pools before metrics."""
    rows = []
    for i, cam in enumerate(cams):
        if args.ssaa > 1:
            cam = dict(cam)
            for k_ in ("fx", "fy", "cx", "cy"):
                cam[k_] = cam[k_] * args.ssaa
            cam["res"] = int(cam["res"] * args.ssaa)
        camt = cam_to_torch(cam, dev)
        if i not in _splat_cache:
            _splat_cache[i] = render_splats(
                g["xyz"], g["opacity"].reshape(-1), g["scale"],
                g["quat"], g["rgb"], camt, return_depth=True)
        s_rgb, s_a, s_d = _splat_cache[i]
        m_rgb, m_a, aux = render_mesh(eval_verts, eval_faces, eval_colors, camt,
                                      return_aux=True)
        if eval_tex is not None:
            res_px = int(camt["res"])
            m_rgb = _tex_sample(aux, eval_tex[0], eval_tex[1]).reshape(res_px, res_px, 3)
        m_d = aux["depth"].reshape(s_d.shape)
        if args.ssaa > 1:
            s_rgb, s_a, s_d = _pool(s_rgb, args.ssaa), _pool(s_a, args.ssaa), _pool(s_d, args.ssaa)
            m_rgb, m_a, m_d = _pool(m_rgb, args.ssaa), _pool(m_a, args.ssaa), _pool(m_d, args.ssaa)
        m = view_metrics(s_rgb, s_a, m_rgb, m_a,
                         splat_depth=s_d, mesh_depth=m_d)
        m.update({"view": i})
        rows.append(m)
        print(f"  [{tag}] view {i}:  psnr={m['psnr']:6.2f}dB  ssim={m['ssim']:.4f}"
              f"  iou={m['iou']:.3f}  d_rmse={m['depth_rmse']:.4f}", flush=True)
        if args.dump:
            from PIL import Image
            from splatforge.render import composite
            for name, rgb_, a_ in (("splat", s_rgb, s_a), (f"mesh_{tag}", m_rgb, m_a)):
                img = (composite(rgb_, a_).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img).save(str(OUT / f"{args.obj}_v{i}_{name}.png"))
            sh = (_shaded(eval_verts, eval_faces, aux, camt).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(sh).save(str(OUT / f"{args.obj}_v{i}_shade_{tag}.png"))
    return {
        "psnr": np.array([r["psnr"] for r in rows]),
        "ssim": np.array([r["ssim"] for r in rows]),
        "iou":  np.array([r["iou"]  for r in rows]),
        "depth_rmse": np.array([r["depth_rmse"] for r in rows]),
    }


def report(tag: str, r: dict):
    print(f"{tag:10s} PSNR {r['psnr'].mean():6.2f} ± {r['psnr'].std():.2f} dB"
          f"   SSIM {r['ssim'].mean():.4f} ± {r['ssim'].std():.4f}"
          f"   IoU {r['iou'].mean():.3f}"
          f"   depth_rmse {r['depth_rmse'].mean():.4f}")


# volsmooth / tsdf: re-extract BOTH sides at full production res (the embedded 50k
# mesh was resampled+decimated, which already hides the artifact frequency bands).
cand_geo = None
if args.candidate == "tsdf":
    from splat_mesh import splat_to_mesh
    from splatforge.tsdf import splat_to_mesh_repair
    print("[tsdf] re-extracting: density std vs TSDF candidate …", flush=True)
    t0 = time.time()
    v0, f0, c0 = splat_to_mesh(g["xyz"], g["opacity"].reshape(-1), scale_raw,
                               g["quat"], g["rgb"], resolution=224, device=dev,
                               taubin=6, vol_smooth=0.7)
    t_std = time.time() - t0
    t0 = time.time()
    v1, f1, c1, _nd = splat_to_mesh_repair(g["xyz"], g["opacity"].reshape(-1), scale_raw,
                                           g["quat"], g["rgb"], resolution=224, device=dev,
                                           taubin=6)
    print(f"[tsdf] std {len(f0)}f ({t_std:.0f}s) / tsdf {len(f1)}f ({time.time()-t0:.0f}s)",
          flush=True)
    verts_np, faces_np = v0, f0
    verts = torch.as_tensor(v0, device=dev)
    faces = torch.as_tensor(f0, device=dev)
    colors = torch.as_tensor(c0, device=dev)
    cams = make_rig(verts_np, k=args.views, res=args.res)
    _splat_cache.clear()
    cand_geo = (torch.as_tensor(v1, device=dev), torch.as_tensor(f1, device=dev),
                torch.as_tensor(c1, device=dev))
elif args.candidate == "volsmooth":
    from splat_mesh import splat_to_mesh
    print("[volsmooth] re-extracting full-res meshes (res=224, taubin=12) …", flush=True)
    t0 = time.time()
    v0, f0, c0 = splat_to_mesh(g["xyz"], g["opacity"].reshape(-1), scale_raw,
                               g["quat"], g["rgb"], resolution=224, device=dev,
                               taubin=12)
    v1, f1, c1 = splat_to_mesh(g["xyz"], g["opacity"].reshape(-1), scale_raw,
                               g["quat"], g["rgb"], resolution=224, device=dev,
                               taubin=12, vol_smooth=args.sigma,
                               vol_smooth_color=args.sigma_color)
    print(f"[volsmooth] std {len(f0)}f / candidate {len(f1)}f  ({time.time()-t0:.1f}s)",
          flush=True)
    verts_np, faces_np = v0, f0
    verts = torch.as_tensor(v0, device=dev)
    faces = torch.as_tensor(f0, device=dev)
    colors = torch.as_tensor(c0, device=dev)
    cams = make_rig(verts_np, k=args.views, res=args.res)
    _splat_cache.clear()
    cand_geo = (torch.as_tensor(v1, device=dev), torch.as_tensor(f1, device=dev),
                torch.as_tensor(c1, device=dev))

print(f"\n--- Standard eval ({args.views} held-out views @ {args.res}px) ---", flush=True)
r_std = evaluate("std", verts, faces, colors)

if args.candidate:
    cand_verts, cand_faces, cand_colors, cand_tex = verts, faces, colors, None
    if args.candidate in ("texture", "texrefine"):
        from texture_bake import bake_texture
        t0 = time.time()
        g_bake = dict(g)
        if not args.bake_smooth:
            g_bake["scale"] = scale_raw   # bake floors 0.25x itself (sharp, noisy)
        # else: keep g's 0.9x-floored scales (smooth, tone-true init)
        nv, nf, uvs_np, tex_np = bake_texture(verts_np, faces_np, g_bake, 2048, dev, False)
        print(f"[texture] 2K bake took {time.time()-t0:.1f}s "
              f"({len(nf)} faces, {len(nv)} verts)", flush=True)
        cand_verts = torch.as_tensor(nv, device=dev)
        cand_faces = torch.as_tensor(nf, device=dev)
        cand_colors = torch.zeros(len(nv), 3, device=dev)   # unused with eval_tex
        uvs_t = torch.as_tensor(uvs_np, device=dev)
        tex_t = torch.as_tensor(tex_np.astype(np.float32) / 255.0, device=dev)
        if args.candidate == "texrefine":
            from splatforge.render.color_opt import refine_texture
            GOLDEN = float(np.pi * (3.0 - np.sqrt(5.0)))
            tr_res = args.train_res or args.res
            train_cams = make_rig(verts_np, k=args.train_views, res=tr_res,
                                  phase=GOLDEN / 2, z_max=0.95)
            t0 = time.time()
            tex_t = refine_texture(cand_verts, cand_faces, uvs_t, tex_t, g, train_cams,
                                   lam_anchor=args.tex_anchor, lam_tv=args.tex_tv)
            print(f"[tex_opt] texel refinement took {time.time()-t0:.1f}s "
                  f"(train_res={tr_res}, anchor={args.tex_anchor}, tv={args.tex_tv})",
                  flush=True)
        cand_tex = (uvs_t, tex_t)
    elif args.candidate == "colors":
        from splatforge.render.color_opt import refine_colors
        GOLDEN = float(np.pi * (3.0 - np.sqrt(5.0)))
        train_cams = make_rig(verts_np, k=args.train_views, res=args.res,
                              phase=GOLDEN / 2, z_max=0.95)
        t0 = time.time()
        cand_colors = refine_colors(verts, faces, colors, g, train_cams)
        print(f"[color_opt] refinement took {time.time()-t0:.1f}s", flush=True)
    elif args.candidate == "smooth":
        from splatforge.smooth import bilateral_smooth
        t0 = time.time()
        cand_verts = torch.as_tensor(
            bilateral_smooth(verts_np, faces_np), device=dev)
        print(f"[smooth] bilateral denoise took {time.time()-t0:.1f}s", flush=True)
    elif args.candidate == "geo":
        from splatforge.render.geo_opt import refine_geometry
        GOLDEN = float(np.pi * (3.0 - np.sqrt(5.0)))
        tr_res = args.train_res or args.res
        train_cams = make_rig(verts_np, k=args.train_views, res=tr_res,
                              phase=GOLDEN / 2, z_max=0.95)
        t0 = time.time()
        cand_verts = refine_geometry(verts, faces, g, train_cams)
        print(f"[geo_opt] DepthFit took {time.time()-t0:.1f}s", flush=True)
    elif args.candidate in ("volsmooth", "tsdf"):
        cand_verts, cand_faces, cand_colors = cand_geo

    print(f"\n--- Candidate eval (same held-out views) ---", flush=True)
    r_cand = evaluate(args.candidate, cand_verts, cand_faces, cand_colors,
                      eval_tex=cand_tex)

    print(f"\n=== PHASE B GATE [{args.obj}]  {args.views} held-out views @ {args.res}px ===")
    report("standard", r_std)
    report(args.candidate, r_cand)
    d_psnr = r_cand["psnr"].mean() - r_std["psnr"].mean()
    d_ssim = r_cand["ssim"].mean() - r_std["ssim"].mean()
    d_rmse = r_cand["depth_rmse"].mean() - r_std["depth_rmse"].mean()
    if args.candidate in ("smooth", "volsmooth", "geo", "tsdf"):
        # geometry candidate: depth fidelity must improve, colour must not regress
        gate = (d_rmse < 0) and (d_psnr > -0.1) and (d_ssim > -0.005)
    else:
        gate = (d_psnr > 0) and (d_ssim > 0)
    print(f"\n[gate]  dPSNR {d_psnr:+.2f} dB   dSSIM {d_ssim:+.4f}   "
          f"d_depth_rmse {d_rmse:+.4f}   "
          f"candidate {'PASSES' if gate else 'FAILS'} on {args.obj}")
else:
    print(f"\n=== PHASE B BASELINE [{args.obj}]  {args.views} views @ {args.res}px ===")
    report("standard", r_std)

if args.dump:
    print(f"PNGs: {OUT}/")

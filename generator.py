"""
TripoSplat extension for Modly.

Reference : https://huggingface.co/spaces/VAST-AI/TripoSplat
Weights   : https://huggingface.co/VAST-AI/TripoSplat

TripoSplat is a feed-forward image-to-3D model that outputs 3D Gaussian Splats.
Modly is mesh-centric (the viewer/exporter only handle .glb), so this extension runs
TripoSplat and extracts a vertex-colored mesh from the Gaussians via splat_mesh.py
(ComfyUI SplatToMesh port: anisotropic density grid + Surface Nets).

The model code (triposplat.py + model.py, pure Python) is bundled in vendor/.
"""
import io
import sys
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from services.generators.base import BaseGenerator, GenerationCancelled

_EXTENSION_DIR = Path(__file__).parent

# Mesh Smoothing label -> Taubin iterations applied during reconstruction. Lower keeps
# crisper relief; the surface stays smooth because Surface Nets already produces a dual
# (non-staircased) surface.
_RECON_SMOOTHING = {"None": 0, "Light": 6, "Medium": 12, "Strong": 20}

# Texture (UI) -> baked-texture resolution. None keeps the vertex-colour path.
# "auto" runs BOTH refined colour paths and keeps the winner of an embedded
# mini-gate (4 held-out views, SSAA, SSIM decides) — no geometric heuristic
# separates the cases reliably (measured), only the empirical comparison does.
_TEX_RES = {"Vertex colors": None, "Baked 1K": 1024, "Baked 2K": 2048, "Auto": "auto"}


class TripoSplatGenerator(BaseGenerator):
    MODEL_ID     = "triposplat"
    DISPLAY_NAME = "TripoSplat"
    VRAM_GB      = 10

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _weights_dir(self) -> Path:
        """Weights live once under the mesh ('generate') node's dir. The splat node
        reuses them (its manifest download_check points there via '..', so Modly does
        not re-download), falling back to its own dir if that shared copy isn't there."""
        if self.model_dir.name == "generate":
            return self.model_dir
        shared = self.model_dir.parent / "generate"
        if (shared / "diffusion_models" / "triposplat_fp16.safetensors").exists():
            return shared
        return self.model_dir

    def is_downloaded(self) -> bool:
        return (self._weights_dir() / "diffusion_models" / "triposplat_fp16.safetensors").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._auto_download()

        self._setup_vendor()

        import torch
        from triposplat import TripoSplatPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"

        md = self._weights_dir()
        print(f"[TripoSplatGenerator] Loading TripoSplat pipeline from {md} …")
        pipe = TripoSplatPipeline(
            ckpt_path              = str(md / "diffusion_models" / "triposplat_fp16.safetensors"),
            decoder_path           = str(md / "vae" / "triposplat_vae_decoder_fp16.safetensors"),
            dinov3_path            = str(md / "clip_vision" / "dino_v3_vit_h.safetensors"),
            flux2_vae_encoder_path = str(md / "vae" / "flux2-vae.safetensors"),
            rmbg_path              = str(md / "background_removal" / "birefnet.safetensors"),
            device                 = device,
        )

        self._model  = pipe
        self._device = device
        print(f"[TripoSplatGenerator] Loaded on {device}.")

    def unload(self) -> None:
        self._device = None
        super().unload()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        if self._is_projection_node():
            return self._run_projection(image_bytes, params, progress_cb, cancel_event)
        return self._run_generate(image_bytes, params, progress_cb, cancel_event)

    def _run_generate(self, image_bytes, params, progress_cb, cancel_event) -> Path:
        steps          = int(params.get("steps", 20))
        num_gaussians  = int(params.get("num_gaussians", 262144))
        mesh_detail    = int(params.get("mesh_detail", 9))
        color_mode     = str(params.get("vivid_color", "Natural"))
        vivid_color    = color_mode == "Vivid"
        color_refine   = color_mode == "Refined"
        taubin         = _RECON_SMOOTHING.get(str(params.get("mesh_smoothing", "Medium")), 12)
        tex_res        = _TEX_RES.get(str(params.get("texture", "Vertex colors")), None)
        seed           = int(params.get("seed", -1))

        # Fixed pipeline values (not user-tunable — kept off the UI as they had no
        # practical effect in this image-to-3D path).
        guidance_scale = 3.0
        shift          = 3.0
        erode_radius   = 1
        if seed < 0:
            seed = int(np.random.randint(0, 2**31 - 1))

        pipe  = self._model
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        self._report(progress_cb, 8, "Generating Gaussians…")

        def _cb(step: int, total: int) -> None:
            if cancel_event and cancel_event.is_set():
                raise GenerationCancelled()
            if progress_cb:
                progress_cb(8 + int(60 * (step / max(1, total))), "Generating Gaussians…")

        gaussian, _prepared = pipe.run(
            image,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            shift=shift,
            num_gaussians=num_gaussians,
            erode_radius=erode_radius,
            callback=_cb,
        )
        self._check_cancelled(cancel_event)

        # Splat node: skip mesh reconstruction, output the raw 3DGS .ply directly.
        if self._is_splat_node():
            return self._export_splat(gaussian, progress_cb)

        self._report(progress_cb, 72, "Reconstructing mesh…")
        mesh = self._reconstruct(gaussian, mesh_detail, vivid_color, taubin,
                                 tex_res, color_refine=color_refine)
        self._check_cancelled(cancel_event)

        ref_pts    = getattr(self, "_bench_ref_pts", None)
        start_mesh = getattr(self, "_bench_start",   None)
        out_path = self._export(mesh, progress_cb)
        self._save_splats(gaussian, out_path, ref_pts=ref_pts, start_mesh=start_mesh)
        return out_path

    def _is_splat_node(self) -> bool:
        # "splat" kept for backward-compat with the pre-rename node id.
        return self.model_dir.name in ("splat", "createSplat")

    def _is_projection_node(self) -> bool:
        return self.model_dir.name == "projection"

    # ------------------------------------------------------------------ #
    # Projection node: image -> in-memory splat -> color a wired grey mesh
    # ------------------------------------------------------------------ #

    def _run_projection(self, image_bytes, params, progress_cb, cancel_event) -> Path:
        """Generate a colored splat from the source image, then project its color
        onto a wired-in grey mesh (the secondary 'mesh' node input). Output is the
        colorized mesh as GLB."""
        import trimesh

        mesh_path = str(params.get("mesh_path", "")).strip()
        if not mesh_path:
            raise ValueError(
                "The Projection node needs a mesh input — wire a mesh node into it."
            )

        steps         = int(params.get("steps", 20))
        num_gaussians = int(params.get("num_gaussians", 262144))
        use_icp       = "ICP" in str(params.get("colorize_mode", "Direct"))
        color_boost   = str(params.get("color_boost", "Vivid + Bright"))
        seed          = int(params.get("seed", -1))
        if seed < 0:
            seed = int(np.random.randint(0, 2**31 - 1))

        # Resolve workspace-relative / "/workspace/..." paths to absolute ones.
        # Same convention as Trellis2's Texture Mesh node: the workflow runner
        # passes a relative path ("Workflows/file.glb"); the Generate page UI
        # passes "/workspace/Default/file.glb". outputs_dir = WORKSPACE_DIR/collection.
        workspace_dir = self.outputs_dir.parent
        mp = Path(mesh_path)
        if mesh_path.startswith("/workspace/"):
            mesh_path = str(workspace_dir / mesh_path[len("/workspace/"):])
        elif not mp.is_absolute():
            mesh_path = str(workspace_dir / mesh_path)

        self._report(progress_cb, 3, "Loading mesh…")
        grey = trimesh.load(mesh_path, force="mesh")
        if not isinstance(grey, trimesh.Trimesh) or len(grey.vertices) == 0:
            raise ValueError(f"Could not load a valid mesh from: {mesh_path}")

        pipe  = self._model
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        self._report(progress_cb, 10, "Generating Gaussians…")

        def _cb(step: int, total: int) -> None:
            if cancel_event and cancel_event.is_set():
                raise GenerationCancelled()
            if progress_cb:
                progress_cb(10 + int(70 * (step / max(1, total))), "Generating Gaussians…")

        gaussian, _prepared = pipe.run(
            image,
            seed=seed,
            steps=steps,
            guidance_scale=3.0,
            shift=3.0,
            num_gaussians=num_gaussians,
            erode_radius=1,
            callback=_cb,
        )
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 86, "Projecting color…")
        grey = self._project_color(grey, gaussian, use_icp, color_boost)
        self._check_cancelled(cancel_event)

        return self._export(grey, progress_cb)

    def _project_color(self, mesh, gaussian, use_icp: bool, color_boost: str = "Natural"):
        """Colorize a grey mesh from the in-memory Gaussian splat. Non-fatal: on
        any failure, log a warning and return the grey mesh unchanged."""
        try:
            if str(_EXTENSION_DIR) not in sys.path:
                sys.path.insert(0, str(_EXTENSION_DIR))
            from colorize import colorize_from_points, _C0

            xyz = gaussian.get_xyz.detach().float().cpu().numpy()
            opacity = gaussian.get_opacity.detach().reshape(-1).float().cpu().numpy()
            rgb = (gaussian._features_dc[:, 0, :].detach().float() * _C0 + 0.5
                   ).clamp(0, 1).cpu().numpy()
            return colorize_from_points(mesh, xyz, rgb, opacity=opacity,
                                        use_icp=use_icp, pre_rotate=True,
                                        color_boost=color_boost)
        except Exception as exc:
            print(f"[TripoSplatGenerator] Projection skipped "
                  f"({'ICP' if use_icp else 'direct'}): {exc}")
            return mesh

    def _export_splat(self, gaussian, progress_cb) -> Path:
        """Splat node output: write the raw 3DGS .ply (the node's output, rendered as a
        smooth Gaussian splat by Modly's viewer) plus a .splat sidecar next to it."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.ply"
        path = self.outputs_dir / name
        gaussian.save_ply(str(path))
        try:
            gaussian.save_splat(str(path.with_suffix(".splat")))
        except Exception as exc:
            print(f"[TripoSplatGenerator] .splat sidecar skipped: {exc}")
        self._report(progress_cb, 100, "Done")
        return path

    @staticmethod
    def _save_splats(gaussian, glb_path: Path, ref_pts=None, start_mesh=None) -> None:
        """Drop the raw Gaussian splat next to the mesh (.splat + 3DGS .ply) and save
        processed Gaussian params + optional ref_pts + bench start mesh as .gaussians.npz.

        start_mesh: (verts, faces) of the Standard mesh in Z-up frame (pre-refine,
        pre-flip). Embedded post-decimation (same 160-res resample + 50k-face prep as
        the texture path) as start_verts/start_faces, making the npz a fully
        self-contained bench input — no GLB filename pairing, no mispair possible.
        Non-fatal — the .glb mesh is the node's actual output."""
        try:
            gaussian.save_splat(str(glb_path.with_suffix(".splat")))
            gaussian.save_ply(str(glb_path.with_suffix(".ply")))
            print(f"[TripoSplatGenerator] Splat preview written next to {glb_path.name} (.splat + .ply)")
        except Exception as exc:
            print(f"[TripoSplatGenerator] Splat export skipped: {exc}")
        try:
            import torch
            import numpy as _np
            if str(_EXTENSION_DIR) not in sys.path:
                sys.path.insert(0, str(_EXTENSION_DIR))
            from splat_mesh import _C0
            _arrays = dict(
                xyz     = gaussian.get_xyz.detach().float().cpu().numpy(),
                opacity = gaussian.get_opacity.detach().reshape(-1).float().cpu().numpy(),
                scale   = torch.sqrt((gaussian.get_scaling.detach().float() ** 2
                                     - gaussian.mininum_kernel_size ** 2).clamp_min(0)).cpu().numpy(),
                quat    = (gaussian._rotation + gaussian.rots_bias[None, :]).detach().float().cpu().numpy(),
                rgb     = (gaussian._features_dc[:, 0, :].detach().float() * _C0 + 0.5).clamp(0, 1).cpu().numpy(),
            )
            if ref_pts is not None:
                _arrays["ref_pts"] = ref_pts
            if start_mesh is not None:
                try:
                    _sv, _sf = TripoSplatGenerator._prep_texture_geometry(
                        start_mesh[0], start_mesh[1], 50_000)
                    _arrays["start_verts"] = _sv.astype(_np.float32)
                    _arrays["start_faces"] = _sf.astype(_np.int32)
                    # Production vertex colours (trilinear volume sampling), remapped
                    # onto the prepped mesh — keeps Phase B renders colour-faithful.
                    if len(start_mesh) > 2 and start_mesh[2] is not None:
                        from scipy.spatial import cKDTree as _KD
                        _, _ci = _KD(start_mesh[0]).query(_sv, k=1)
                        _arrays["start_colors"] = (
                            _np.clip(start_mesh[2][_ci], 0, 1) * 255).astype(_np.uint8)
                except Exception as _exc:
                    print(f"[TripoSplatGenerator] start mesh embed skipped: {_exc}", flush=True)
            _np.savez_compressed(str(glb_path.with_suffix(".gaussians.npz")), **_arrays)
            _ref_info = f"  ref_pts={len(ref_pts)}" if ref_pts is not None else ""
            if "start_verts" in _arrays:
                _ref_info += f"  start_mesh={len(_arrays['start_faces'])}f"
            print(f"[TripoSplatGenerator] Gaussians saved: {glb_path.stem}.gaussians.npz{_ref_info}")
        except Exception as exc:
            print(f"[TripoSplatGenerator] Gaussians npz save skipped: {exc}")

    # Mesh Detail (UI) -> Surface-Nets density-grid resolution. The spacing-based
    # scale floor keeps the surface solid, so very high res mostly adds cost.
    _RES = {7: 160, 8: 192, 9: 224, 10: 288}

    def _reconstruct(self, gaussian, mesh_detail: int, vivid_color: bool = False, taubin: int = 6,
                     tex_res=None, color_refine: bool = False):
        """Gaussian splat -> vertex-colored mesh via the ported ComfyUI SplatToMesh
        (anisotropic density grid + Surface Nets). Falls back to CPU on CUDA OOM."""
        # Texture "Auto": run the vertex path with refined colours, build the
        # refined-texture candidate too, and keep the winner of the embedded
        # mini-gate. Implies render-fitted (Natural-look) colours.
        tex_auto = (tex_res == "auto")
        if tex_auto:
            tex_res = None
            color_refine = True
        import torch
        import trimesh
        if str(_EXTENSION_DIR) not in sys.path:
            sys.path.insert(0, str(_EXTENSION_DIR))
        from splat_mesh import splat_to_mesh, _C0

        # Cast to fp32: the model runs in fp16, and the covariance math (1/scale^2,
        # exp) is unstable / overflow-prone in half precision (as ComfyUI does too).
        xyz     = gaussian.get_xyz.detach().float()
        scale   = gaussian.get_scaling.detach().float()
        opacity = gaussian.get_opacity.detach().reshape(-1).float()
        quat    = (gaussian._rotation + gaussian.rots_bias[None, :]).detach().float()
        rgb     = (gaussian._features_dc[:, 0, :].detach().float() * _C0 + 0.5).clamp(0, 1)

        res = self._RES.get(int(mesh_detail), 224)
        dev = xyz.device

        # vol_smooth=0.7: ~1-voxel blur of the density+colour grids before Surface
        # Nets — removes the inter-gaussian "crevasse" valleys at the source.
        # Quality modes (Refined / Auto) use the defect-driven repair extraction
        # instead: same density surface, plus detection of sunken pockets / true
        # holes (mesh depth vs splat expected depth on 24 views) repaired by a
        # masked TSDF fusion. Gate 4/4: +0.9 dB on a shredded splat, crater on
        # the back of heads fixed, BIT-IDENTICAL no-op on defect-free objects.
        def _run(d):
            # The defect-driven repair extraction now runs in EVERY mode (un-gated
            # 2026-06-20), not just Refined/Auto: it is a bit-identical no-op on
            # defect-free objects and fixes blind-spot craters (e.g. the cat's
            # shredded head crown) that the fast Vertex/Natural path otherwise ships
            # holey. DepthFit + colour-refine stay gated below (cost / hard-edge risk).
            try:
                if str(_EXTENSION_DIR) not in sys.path:
                    sys.path.insert(0, str(_EXTENSION_DIR))
                from splatforge.tsdf import splat_to_mesh_repair
                out = splat_to_mesh_repair(
                    xyz.to(d), opacity.to(d), scale.to(d), quat.to(d),
                    rgb.to(d), resolution=res, device=d, taubin=taubin)
                if out is not None:
                    return out[0], out[1], out[2]
            except Exception as exc:
                print(f"[TripoSplatGenerator] repair extraction failed ({exc}); "
                      f"using standard extraction.", flush=True)
            return splat_to_mesh(xyz.to(d), opacity.to(d), scale.to(d), quat.to(d),
                                 rgb.to(d), resolution=res, device=d,
                                 vivid_color=vivid_color, taubin=taubin,
                                 vol_smooth=0.7)
        try:
            out = _run(dev)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            cpu = torch.device("cpu")
            out = _run(cpu)

        if out is None:
            raise RuntimeError("SplatToMesh produced no surface (empty splat / level off-histogram).")
        verts, faces, colors = out

        if len(faces) > 0:
            verts, faces, colors = self._close_holes(verts, faces, colors)

        # Vertex budget: the geomean voxel grid hands elongated objects their full
        # res^3 budget, which at Ultra can mean 1.8-2.6M faces — every downstream
        # refine scales with it and the GLB balloons. Colour-aware decimation to
        # 1M is render-indistinguishable (measured: truck 2.55M -> 1M, SSIM
        # 0.8865 -> 0.8866 at 512, identical at 128; cat -0.001) because vertex
        # quality = local colour gradient protects detail (eyes, patterns) while
        # flat regions collapse. ~20 s, repaid 2-3x by the refines.
        if len(faces) > 1_200_000:
            try:
                verts, faces, colors = self._decimate_color_budget(
                    verts, faces, colors, 1_000_000)
            except Exception as exc:
                print(f"[TripoSplatGenerator] budget decimation skipped ({exc})",
                      flush=True)

        # Sample reference point cloud from the filled high-res mesh (before any decimation).
        # Saved to .gaussians.npz as 'ref_pts' for chamfer-distance bench (independent fidelity).
        # Also stash the Standard mesh (Z-up, pre-refine) so _save_splats can embed the
        # bench start mesh in the npz — self-contained input, no GLB filename pairing.
        self._bench_ref_pts = None
        self._bench_start   = None
        if len(faces) > 0:
            try:
                import trimesh as _tm
                _rm = _tm.Trimesh(vertices=verts, faces=faces, process=False)
                _rp, _ = _tm.sample.sample_surface(_rm, 500_000)
                self._bench_ref_pts = _rp.astype(np.float32)
                self._bench_start   = (np.asarray(verts, np.float32).copy(),
                                       np.asarray(faces, np.int64).copy(),
                                       np.asarray(colors, np.float32).copy())
            except Exception as _exc:
                print(f"[TripoSplatGenerator] ref_pts sampling skipped: {_exc}", flush=True)

        # DepthFit (quality modes): depth-supervised linearized geometry refinement —
        # vertices move along their normals so the rasterized depth matches the
        # splat's expected depth (smooth), flattening the lumpy relief that ruins
        # the lit appearance. Helps organic/damaged surfaces (+0.2..+3.1 dB), hurts
        # hard-edged objects (−2 dB on the wooden car) → guarded per object on
        # fresh validation views, kept only if depth fidelity improves. Non-fatal.
        if (color_refine or tex_auto) and len(faces) > 0:
            try:
                if str(_EXTENSION_DIR) not in sys.path:
                    sys.path.insert(0, str(_EXTENSION_DIR))
                from scipy.spatial import cKDTree as _KD
                from splatforge.render.geo_opt import refine_geometry
                from splatforge.render.auto import guard_geometry
                from splatforge.render.cameras import make_rig as _mk_rig
                keep = opacity >= 0.02
                g_geo = {"xyz": xyz[keep].to(dev), "opacity": opacity[keep].to(dev),
                         "scale": scale[keep].to(dev), "quat": quat[keep].to(dev),
                         "rgb": rgb[keep].to(dev)}
                _xn = g_geo["xyz"].cpu().numpy()
                _sb = _xn[np.random.default_rng(0).choice(
                    len(_xn), min(20000, len(_xn)), replace=False)]
                _sp = float(np.median(_KD(_xn).query(_sb, k=2)[0][:, 1]))
                g_geo["scale"] = g_geo["scale"].clamp_min(0.9 * _sp)
                _GOLD = float(np.pi * (3.0 - np.sqrt(5.0)))
                _cams = _mk_rig(np.asarray(verts, np.float32), k=16, res=512,
                                phase=_GOLD / 2, z_max=0.95)
                print("[TripoSplatGenerator] DepthFit (depth-supervised geometry) …", flush=True)
                _t0 = time.time()
                _vt = torch.as_tensor(np.asarray(verts, np.float32), device=dev)
                _ft = torch.as_tensor(np.asarray(faces, np.int64), device=dev)
                _v2 = refine_geometry(_vt, _ft, g_geo, _cams)
                if guard_geometry(_vt, _v2, _ft, g_geo, dev):
                    verts = _v2.cpu().numpy().astype(np.float32)
                    print(f"[TripoSplatGenerator] DepthFit accepted ({time.time()-_t0:.1f}s)", flush=True)
                else:
                    print(f"[TripoSplatGenerator] DepthFit rejected by guard "
                          f"({time.time()-_t0:.1f}s) — keeping extraction geometry.", flush=True)
            except Exception as exc:
                print(f"[TripoSplatGenerator] DepthFit skipped ({exc})", flush=True)

        # Render-supervised colour refinement (Color Mode: Refined) — fits vertex
        # colours to multi-view splat renders (Phase B gate: PSNR/SSIM up on all test
        # objects vs trilinear sampling). Z-up frame required (renders the gaussians),
        # so this runs BEFORE the flip. Vertex-colour path only: the baked-texture
        # path samples its texels straight from the gaussians. Non-fatal.
        if color_refine and not tex_res and len(faces) > 0:
            try:
                if str(_EXTENSION_DIR) not in sys.path:
                    sys.path.insert(0, str(_EXTENSION_DIR))
                from splatforge.render.color_opt import refine_colors_production
                keep = opacity >= 0.02
                g_cr = {"xyz": xyz[keep], "opacity": opacity[keep], "scale": scale[keep],
                        "quat": quat[keep], "rgb": rgb[keep]}
                print("[TripoSplatGenerator] Color refine (render-supervised) …", flush=True)
                _t0 = time.time()
                colors = refine_colors_production(verts, faces, colors, g_cr, dev)
                print(f"[TripoSplatGenerator] Color refine done ({time.time()-_t0:.1f}s)", flush=True)
            except Exception as exc:
                print(f"[TripoSplatGenerator] Color refine skipped ({exc}); "
                      f"keeping volume-sampled colors.", flush=True)

        # Splat frame (Z-up) -> Modly/glTF (Y-up): stand the model upright instead of
        # lying on its side. Proper rotation (det +1) — winding and colors unchanged.
        _FLIP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)

        # Texture "Auto": build the refined-texture candidate and let the embedded
        # mini-gate pick vertex vs texture (4 fresh held-out views, SSAA, SSIM
        # decides). Validated 4/4 against the independent bench verdicts.
        # Non-fatal: any failure keeps the (already refined) vertex path.
        if tex_auto and len(faces) > 0:
            try:
                if str(_EXTENSION_DIR) not in sys.path:
                    sys.path.insert(0, str(_EXTENSION_DIR))
                from scipy.spatial import cKDTree as _KD
                from splatforge.render.auto import pick_color_path
                print("[TripoSplatGenerator] Auto texture: baking candidate …", flush=True)
                nv, nf, uvs, tex, g_auto = self._bake_candidate(
                    verts, faces, xyz, opacity, scale, quat, rgb, 2048, dev,
                    False, color_refine=True)
                # floored copy: the validation splat must render the same field
                # the meshes were extracted from
                _xn = g_auto["xyz"].detach().float().cpu().numpy()
                _sb = _xn[np.random.default_rng(0).choice(
                    len(_xn), min(20000, len(_xn)), replace=False)]
                _sp = float(np.median(_KD(_xn).query(_sb, k=2)[0][:, 1]))
                g_val = {k: v.detach().float().to(dev) for k, v in g_auto.items()}
                g_val["scale"] = g_val["scale"].clamp_min(0.9 * _sp)
                r = pick_color_path((verts, faces, colors), (nv, nf, uvs, tex),
                                    g_val, dev)
                if r["winner"] == "texture":
                    return self._build_textured_mesh(nv, nf, uvs, tex, _FLIP)
                # else fall through to the vertex-colour path below
            except Exception as exc:
                print(f"[TripoSplatGenerator] Auto texture skipped ({exc}); "
                      f"using vertex colors.", flush=True)

        # Baked-texture path: UV-unwrap and sample colour straight from the Gaussians
        # (no voxel grid). Bakes in splat space, then applies the SAME Y/Z flip. On any
        # failure (e.g. the xatlas unwrap crashing) fall through to the vertex-colour
        # path below — colors/verts/faces are still in scope, so the run still produces
        # a valid GLB instead of erroring out.
        if tex_res:
            try:
                return self._reconstruct_textured(
                    verts, faces, xyz, opacity, scale, quat, rgb, tex_res, dev,
                    vivid_color, _FLIP, color_refine=color_refine)
            except Exception as exc:
                print(f"[TripoSplatGenerator] Texture bake failed ({exc}); "
                      f"falling back to vertex colors.")

        verts = verts @ _FLIP

        # Hard safety cap on triangle count; re-map colors onto the decimated verts.
        _CAP = 3000000
        if len(faces) > _CAP:
            mesh = self._decimate(trimesh.Trimesh(vertices=verts, faces=faces, process=False), _CAP)
            from scipy.spatial import cKDTree
            _, idx = cKDTree(verts).query(mesh.vertices, k=1)
            colors = colors[idx]
        else:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        mesh.visual.vertex_colors = np.concatenate(
            [(np.clip(colors, 0, 1) * 255).astype(np.uint8),
             np.full((len(colors), 1), 255, dtype=np.uint8)], axis=1)
        return mesh

    def _bake_candidate(self, verts, faces, xyz, opacity, scale, quat, rgb,
                        tex_res, dev, vivid_color, color_refine):
        """Prep + unwrap + bake (+ optional texel refine). Returns
        (nv, nf, uvs, tex_uint8, g) in splat/Z-up space — no flip, no trimesh."""
        import torch
        from texture_bake import bake_texture

        # Close + clean the surface, then decimate to a budget xatlas unwraps quickly and
        # safely on Windows. The raw Surface-Nets surface is non-watertight with
        # non-manifold edges; decimating it directly TEARS those micro-gaps into large
        # visible holes (and on dirty topology xatlas is ~10x slower / crashes sooner).
        # Uniform volumetric resampling first yields a closed manifold, so decimation
        # stays hole-free AND the unwrap is faster. The texture samples Gaussians per
        # texel, so colour is unaffected by the lower poly.
        _TEX_FACE_CAP = 50000
        verts, faces = self._prep_texture_geometry(verts, faces, _TEX_FACE_CAP)

        keep = opacity >= 0.02
        g = {"xyz": xyz[keep], "opacity": opacity[keep], "scale": scale[keep],
             "quat": quat[keep], "rgb": rgb[keep]}
        try:
            nv, nf, uvs, tex = bake_texture(verts, faces, g, tex_res, dev, vivid_color)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            cpu = torch.device("cpu")
            g = {k: v.to(cpu) for k, v in g.items()}
            nv, nf, uvs, tex = bake_texture(verts, faces, g, tex_res, cpu, vivid_color)

        # Color Mode "Refined": fit the baked texels to multi-view splat renders.
        # Corrects the per-texel KNN estimator's tone bias while keeping its
        # sharpness. Z-up frame required — runs before the flip. Non-fatal.
        if color_refine:
            try:
                if str(_EXTENSION_DIR) not in sys.path:
                    sys.path.insert(0, str(_EXTENSION_DIR))
                from splatforge.render.color_opt import refine_texture_production
                print("[TripoSplatGenerator] Texture refine (render-supervised) …", flush=True)
                _t0 = time.time()
                tex = refine_texture_production(nv, nf, uvs, tex, g, dev)
                print(f"[TripoSplatGenerator] Texture refine done ({time.time()-_t0:.1f}s)",
                      flush=True)
            except Exception as exc:
                print(f"[TripoSplatGenerator] Texture refine skipped ({exc}); "
                      f"keeping raw bake.", flush=True)
        return nv, nf, uvs, tex, g

    @staticmethod
    def _build_textured_mesh(nv, nf, uvs, tex, flip):
        """Z-up baked candidate -> Y-up textured trimesh (glTF-ready)."""
        import trimesh
        from PIL import Image
        from trimesh.visual import TextureVisuals
        from trimesh.visual.material import PBRMaterial
        nv = (nv @ flip).astype(np.float32)
        img = Image.fromarray(tex)
        material = PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=1.0)
        return trimesh.Trimesh(vertices=nv, faces=nf,
                               visual=TextureVisuals(uv=uvs, material=material),
                               process=False)

    def _reconstruct_textured(self, verts, faces, xyz, opacity, scale, quat, rgb,
                              tex_res, dev, vivid_color, flip, color_refine=False):
        """Baked-texture branch: bake candidate + flip + self-contained textured GLB."""
        nv, nf, uvs, tex, _ = self._bake_candidate(
            verts, faces, xyz, opacity, scale, quat, rgb, tex_res, dev,
            vivid_color, color_refine)
        return self._build_textured_mesh(nv, nf, uvs, tex, flip)

    # Volumetric-resample grid resolution for the texture path. Fixed (not tied to Mesh
    # Detail): it sets the closed-surface silhouette detail AND drives xatlas unwrap time
    # (160 -> ~17s unwrap / ~20s total bake; 224 -> ~29s). The 2K texture carries the
    # colour detail, so a moderate silhouette resolution is the right speed/quality point.
    _TEX_REMESH_RES = 160

    @staticmethod
    def _prep_texture_geometry(verts, faces, target):
        """Close the dual-contour surface (uniform volumetric resampling -> a closed
        manifold, removing the micro-holes/non-manifold edges that would otherwise tear
        open under decimation) then decimate to `target` faces for a fast, hole-free
        xatlas unwrap. Falls back to plain decimation if resampling fails. pymeshlab is
        already a dependency; this avoids the skimage that trimesh's voxel path needs."""
        import pymeshlab as ml
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(vertex_matrix=verts.astype(np.float64),
                            face_matrix=faces.astype(np.int32)))
        try:
            ms.generate_resampled_uniform_mesh(
                cellsize=ml.PercentageValue(100.0 / TripoSplatGenerator._TEX_REMESH_RES),
                offset=ml.PercentageValue(50.0), mergeclosevert=True)
        except Exception as exc:
            print(f"[TripoSplatGenerator] uniform remesh skipped ({exc}); decimating raw surface.")
        if ms.current_mesh().face_number() > int(target):
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), preservetopology=True)
        cm = ms.current_mesh()
        return (np.ascontiguousarray(cm.vertex_matrix(), np.float32),
                np.ascontiguousarray(cm.face_matrix(), np.int64))

    @staticmethod
    def _close_holes(verts, faces, colors, maxholesize: int = 1000000):
        """Cap open boundary holes so the mesh is watertight — you no longer see through
        to the hollow/dark interior, including the big opening under the base. The size
        limit is effectively unlimited so even large openings get capped. pymeshlab
        triangulates existing boundary loops; colors are re-mapped onto the result by
        nearest original vertex (robust even if the vertex set changes)."""
        import pymeshlab as ml
        from scipy.spatial import cKDTree
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(vertex_matrix=verts.astype(np.float64),
                            face_matrix=faces.astype(np.int32)))
        # NOTE: do NOT call meshing_repair_non_manifold_edges() here — on a dual-contour
        # surface it splits non-manifold edges into thousands of new open boundaries
        # (measured: 0 -> 30k boundary edges) that close_holes then can't fully cap,
        # leaving the mesh worse (full of real holes). The raw surface already has ~0
        # open boundaries, so a plain close_holes is a safe near-no-op / small-hole cap.
        try:
            ms.meshing_close_holes(maxholesize=int(maxholesize))
        except Exception as exc:
            print(f"[TripoSplatGenerator] close_holes skipped: {exc}")
            return verts, faces, colors
        m = ms.current_mesh()
        v2 = m.vertex_matrix().astype(np.float32)
        f2 = m.face_matrix().astype(np.int64)
        _, idx = cKDTree(verts).query(v2, k=1)
        return v2, f2, colors[idx]

    @staticmethod
    def _decimate_color_budget(verts, faces, colors, target: int):
        """Quadric decimation weighted by the local colour gradient: vertex
        quality = mean |dRGB| over incident edges (normalized at p95, weight up
        to 10x), so colour detail survives while uniform regions collapse.
        Colours are re-mapped by nearest original vertex."""
        import pymeshlab as ml
        from scipy.spatial import cKDTree
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
        dc = np.linalg.norm(colors[e[:, 0]] - colors[e[:, 1]], axis=1)
        grad = np.zeros(len(verts), np.float64)
        cnt = np.zeros(len(verts), np.float64)
        np.add.at(grad, e[:, 0], dc)
        np.add.at(grad, e[:, 1], dc)
        np.add.at(cnt, e[:, 0], 1)
        np.add.at(cnt, e[:, 1], 1)
        grad /= np.maximum(cnt, 1)
        q = np.clip(1.0 + 9.0 * grad / max(np.percentile(grad, 95), 1e-9), 1.0, 10.0)
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(vertex_matrix=verts.astype(np.float64),
                            face_matrix=faces.astype(np.int32),
                            v_scalar_array=q))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), preservetopology=True, qualityweight=True)
        cm = ms.current_mesh()
        nv = np.ascontiguousarray(cm.vertex_matrix(), np.float32)
        nf = np.ascontiguousarray(cm.face_matrix(), np.int64)
        _, idx = cKDTree(verts).query(nv, k=1)
        print(f"[TripoSplatGenerator] vertex budget: {len(faces)} -> {len(nf)} faces "
              f"(colour-aware)", flush=True)
        return nv, nf, colors[idx]

    @staticmethod
    def _decimate(mesh, target: int):
        """Quadric edge-collapse decimation via pymeshlab (already a dependency)."""
        import pymeshlab as ml
        import trimesh
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(vertex_matrix=mesh.vertices.astype(np.float64),
                            face_matrix=mesh.faces.astype(np.int32)))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target))
        m = ms.current_mesh()
        return trimesh.Trimesh(vertices=m.vertex_matrix().astype(np.float32),
                               faces=m.face_matrix().astype(np.int64), process=False)

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _smooth_vertex_normals(mesh):
        """Feature-preserving (BILATERAL) smoothed vertex normals, positions untouched.

        The area-weighted normals of a high-res Surface-Nets mesh carry the grid's
        sub-voxel relief noise. UNLIT that is invisible (flat DC colour), but the
        Modly viewer LIGHTS the mesh from the stored normals, so the noise shades
        as mottling/orange-peel — the "c'est moche" complaint. A bilateral normal
        filter (neighbour weight by normal similarity) kills that high-frequency
        noise while PRESERVING genuine creases/features — a uniform Laplacian smears
        them (the see-saw: cleaning the duck smeared the cat's muzzle). Shared with
        the LIT regression harness (debug/lit_regression.py) so what ships is what
        was validated. Bilateral >= Laplacian on all 9 test objects (LIT)."""
        from splatforge.normals import smooth_normals
        return smooth_normals(np.asarray(mesh.vertices, np.float32),
                              np.asarray(mesh.faces, np.int64),
                              normals0=np.asarray(mesh.vertex_normals, np.float32),
                              iters=20, mode="bilateral", sigma=0.30)

    def _export(self, mesh, progress_cb) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        # Write SMOOTHED vertex normals so trimesh stores the NORMAL attribute.
        # Without explicit normals the viewer computes flat per-face ones (noisy
        # faceting); with the raw area-weighted ones the grid relief shades as
        # mottling under the viewer's lighting. Smoothing the normals fixes both.
        try:
            mesh.vertex_normals = self._smooth_vertex_normals(mesh)
        except Exception as exc:
            print(f"[TripoSplatGenerator] normal smoothing skipped ({exc})", flush=True)
            try:
                _ = mesh.vertex_normals
            except Exception:
                pass
        mesh.export(str(path))
        self._finalize_glb(path)
        self._report(progress_cb, 100, "Done")
        return path

    @staticmethod
    def _finalize_glb(path: Path) -> None:
        """Ensure every primitive has an explicit non-metallic material.

        trimesh's vertex-colour export writes NO material, and the glTF spec
        default is metallicFactor=1.0 — a metal: diffuse goes black and the
        surface mostly mirrors the environment, which reads as a dark, crusty,
        dirty look under the viewer's lighting (worst on bumpy regions). The
        baked-texture path already sets metallic=0 explicitly; this brings the
        vertex-colour path in line. Pure JSON-chunk edit, non-fatal."""
        import json, struct
        try:
            raw = Path(path).read_bytes()
            magic, version, _ = struct.unpack_from("<III", raw, 0)
            if magic != 0x46546C67:                      # b'glTF'
                return
            jlen, jtype = struct.unpack_from("<II", raw, 12)
            if jtype != 0x4E4F534A:                      # b'JSON'
                return
            js = json.loads(raw[20:20 + jlen])
            if js.get("materials"):
                return                                   # already has materials
            js["materials"] = [{
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": False,
            }]
            for mesh_def in js.get("meshes", []):
                for prim in mesh_def.get("primitives", []):
                    prim.setdefault("material", 0)
            jbytes = json.dumps(js, separators=(",", ":")).encode("utf-8")
            jbytes += b" " * (-len(jbytes) % 4)          # 4-byte alignment (spaces)
            rest = raw[20 + jlen:]                       # BIN chunk(s), already aligned
            total = 12 + 8 + len(jbytes) + len(rest)
            out = struct.pack("<III", magic, version, total) \
                + struct.pack("<II", len(jbytes), jtype) + jbytes + rest
            Path(path).write_bytes(out)
        except Exception as exc:
            print(f"[TripoSplatGenerator] GLB material finalize skipped: {exc}", flush=True)

    def _setup_vendor(self) -> None:
        import torch  # noqa: F401  (registers DLL dir on Windows before model import)
        vendor_dir = _EXTENSION_DIR / "vendor"
        if vendor_dir.exists() and str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
        try:
            from triposplat import TripoSplatPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "[TripoSplatGenerator] triposplat module not found. Run build_vendor.py "
                "with the app's Python, or click Repair on the Models page."
            ) from exc

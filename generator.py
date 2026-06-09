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
        vivid_color    = str(params.get("vivid_color", "Natural")) == "Vivid"
        taubin         = _RECON_SMOOTHING.get(str(params.get("mesh_smoothing", "Medium")), 12)
        fill_mode      = str(params.get("fill_holes", "On"))
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
        mesh = self._reconstruct(gaussian, mesh_detail, vivid_color, taubin, fill_mode)
        self._check_cancelled(cancel_event)

        out_path = self._export(mesh, progress_cb)
        self._save_splats(gaussian, out_path)
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
    def _save_splats(gaussian, glb_path: Path) -> None:
        """Also drop the raw Gaussian splat next to the mesh (.splat + 3DGS .ply) so it
        can be imported into Modly's native splat viewer (full splat fidelity, no mesh
        approximation). Non-fatal — the .glb mesh is the node's actual output."""
        try:
            gaussian.save_splat(str(glb_path.with_suffix(".splat")))
            gaussian.save_ply(str(glb_path.with_suffix(".ply")))
            print(f"[TripoSplatGenerator] Splat preview written next to {glb_path.name} (.splat + .ply)")
        except Exception as exc:
            print(f"[TripoSplatGenerator] Splat export skipped: {exc}")

    # Mesh Detail (UI) -> Surface-Nets density-grid resolution. The spacing-based
    # scale floor keeps the surface solid, so very high res mostly adds cost.
    _RES = {7: 160, 8: 192, 9: 224, 10: 288}

    def _reconstruct(self, gaussian, mesh_detail: int, vivid_color: bool = False, taubin: int = 6,
                     fill_mode: str = "On"):
        """Gaussian splat -> vertex-colored mesh via the ported ComfyUI SplatToMesh
        (anisotropic density grid + Surface Nets). Falls back to CPU on CUDA OOM."""
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

        def _run(d):
            return splat_to_mesh(xyz.to(d), opacity.to(d), scale.to(d), quat.to(d),
                                 rgb.to(d), resolution=res, device=d,
                                 vivid_color=vivid_color, taubin=taubin)
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
            if fill_mode == "Solid":
                verts, faces, colors = self._solidify(verts, faces, colors, res)
            elif fill_mode != "Off":
                verts, faces, colors = self._close_holes(verts, faces, colors)

        # Splat frame (Z-up) -> Modly/glTF (Y-up): stand the model upright instead of
        # lying on its side. Proper rotation (det +1) — winding and colors unchanged.
        verts = verts @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)

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
    def _solidify(verts, faces, colors, resolution):
        """Guaranteed-watertight 'nuclear' fill: voxelize the mesh, flood-fill the
        interior to a solid, and re-extract a closed surface (marching cubes). Removes
        every hole AND the hollow interior, at the cost of some fine detail. The voxel
        grid is mapped back to world space (`vg.transform`), a light Taubin pass removes
        the staircase, and colors are re-mapped by nearest original vertex. Falls back to
        hole-capping if voxelization fails."""
        import trimesh
        from scipy.spatial import cKDTree
        try:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
            pitch = diag / float(min(int(resolution), 256))
            vg = mesh.voxelized(pitch=pitch).fill()
            out = vg.marching_cubes
            out.apply_transform(vg.transform)
            out.update_faces(out.unique_faces())
            out.remove_unreferenced_vertices()
            trimesh.smoothing.filter_taubin(out, iterations=10)
            v2 = out.vertices.astype(np.float32)
            f2 = out.faces.astype(np.int64)
        except Exception as exc:
            print(f"[TripoSplatGenerator] solidify failed ({exc}); falling back to hole-cap.")
            return TripoSplatGenerator._close_holes(verts, faces, colors)
        _, idx = cKDTree(verts).query(v2, k=1)
        return v2, f2, colors[idx]

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

    def _export(self, mesh, progress_cb) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))
        self._report(progress_cb, 100, "Done")
        return path

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

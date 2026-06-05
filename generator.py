"""
TripoSplat extension for Modly.

Reference : https://huggingface.co/spaces/VAST-AI/TripoSplat
Weights   : https://huggingface.co/VAST-AI/TripoSplat

TripoSplat is a feed-forward image-to-3D model that outputs 3D Gaussian Splats.
Modly is mesh-centric (the viewer/exporter only handle .glb), so this extension
turns the Gaussians into a mesh, split across two nodes:

  • generate (image -> mesh): run TripoSplat, reconstruct the Gaussians into a
    vertex-colored Poisson mesh at full density.
  • texture  (image+mesh -> mesh): post-process any mesh — smooth it, optionally
    decimate, UV-unwrap it and bake the vertex colors into a texture atlas. This
    is where the "nice" surface color comes from (a texture decouples color from
    triangle count, unlike per-vertex color).

The model code (triposplat.py + model.py, pure Python) is bundled in vendor/.
"""
import io
import os
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

# SH degree-0 -> RGB constant (standard 3D Gaussian Splatting convention)
_SH_C0 = 0.28209479177387814

# Surface Smoothing label -> Taubin iterations (texture node)
_SMOOTHING = {"None": 0, "Light": 10, "Medium": 25, "Strong": 50}


class TripoSplatGenerator(BaseGenerator):
    MODEL_ID     = "triposplat"
    DISPLAY_NAME = "TripoSplat"
    VRAM_GB      = 10

    def _node(self) -> str:
        """Active node id ('generate' or 'texture'), taken from the model dir."""
        return self.model_dir.name

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        if self._node() == "texture":
            return True  # mesh post-processing — no model weights needed
        return (self.model_dir / "diffusion_models" / "triposplat_fp16.safetensors").exists()

    def load(self) -> None:
        if self._node() == "texture":
            self._model = self._model or "texture-node"  # no heavy model to load
            return

        if self._model is not None and self._model != "texture-node":
            return

        if not self.is_downloaded():
            self._auto_download()

        self._setup_vendor()

        import torch
        from triposplat import TripoSplatPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[TripoSplatGenerator] Loading TripoSplat pipeline from {self.model_dir} …")
        pipe = TripoSplatPipeline(
            ckpt_path              = str(self.model_dir / "diffusion_models" / "triposplat_fp16.safetensors"),
            decoder_path           = str(self.model_dir / "vae" / "triposplat_vae_decoder_fp16.safetensors"),
            dinov3_path            = str(self.model_dir / "clip_vision" / "dino_v3_vit_h.safetensors"),
            flux2_vae_encoder_path = str(self.model_dir / "vae" / "flux2-vae.safetensors"),
            rmbg_path              = str(self.model_dir / "background_removal" / "birefnet.safetensors"),
            device                 = device,
        )

        self._model  = pipe
        self._device = device
        print(f"[TripoSplatGenerator] Loaded on {device}.")

    def unload(self) -> None:
        self._device = None
        super().unload()

    # ------------------------------------------------------------------ #
    # Inference dispatch
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        if self._node() == "texture":
            return self._run_texture(params, progress_cb, cancel_event)
        return self._run_generate(image_bytes, params, progress_cb, cancel_event)

    # ------------------------------------------------------------------ #
    # Node 1 — generate (image -> vertex-colored mesh)
    # ------------------------------------------------------------------ #

    def _run_generate(self, image_bytes, params, progress_cb, cancel_event) -> Path:
        steps          = int(params.get("steps", 20))
        guidance_scale = float(params.get("guidance_scale", 3.0))
        num_gaussians  = int(params.get("num_gaussians", 262144))
        mesh_detail    = int(params.get("mesh_detail", 9))
        shift          = float(params.get("shift", 3.0))
        erode_radius   = int(params.get("erode_radius", 1))
        seed           = int(params.get("seed", -1))
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

        self._report(progress_cb, 72, "Reconstructing mesh…")
        xyz, rgb = self._gaussian_arrays(gaussian)
        mesh = self._reconstruct_vertex_colored(xyz, rgb, depth=mesh_detail)
        self._check_cancelled(cancel_event)

        return self._export(mesh, progress_cb)

    def _reconstruct_vertex_colored(self, xyz: np.ndarray, rgb: np.ndarray, depth: int):
        """Poisson reconstruction (pymeshlab) at full density with smoothed
        per-vertex colors. Kept dense on purpose so the texture node has a rich
        color source to bake from."""
        import pymeshlab as ml
        import trimesh
        from scipy.spatial import cKDTree

        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(vertex_matrix=xyz.astype(np.float64)), "points")
        ms.compute_normal_for_point_clouds(k=10)
        ms.generate_surface_reconstruction_screened_poisson(depth=int(depth), preclean=True)
        try:
            ms.apply_coord_taubin_smoothing(stepsmoothnum=8)
        except Exception as exc:
            print(f"[TripoSplatGenerator] Taubin smoothing skipped: {exc}")
        try:
            ms.meshing_remove_connected_component_by_face_number(mincomponentsize=5000)
        except Exception as exc:
            print(f"[TripoSplatGenerator] Component cleanup skipped: {exc}")

        # Cap density: a high Mesh Detail (depth 10) Poisson can exceed 1M faces,
        # which makes the downstream Texture node's UV/bake crawl. Stay rich but
        # bounded — this is still a fine color source to bake from.
        _CAP = 300000
        if ms.current_mesh().face_number() > _CAP:
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=_CAP)

        m     = ms.current_mesh()
        verts = m.vertex_matrix().astype(np.float32)
        faces = m.face_matrix().astype(np.int64)
        if verts.size == 0 or faces.size == 0:
            raise RuntimeError("Poisson reconstruction produced an empty mesh.")

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # Colour: inverse-distance average of the k nearest Gaussians, then a
        # couple of Laplacian passes — kills the salt-and-pepper of a 1-NN map.
        k = min(16, len(xyz))
        dist, idx = cKDTree(xyz).query(verts, k=k)
        w   = (1.0 / np.maximum(dist, 1e-6))[..., None]
        col = (rgb[idx] * w).sum(axis=1) / w.sum(axis=1)
        col = self._smooth_colors(np.clip(col, 0.0, 1.0).astype(np.float32), mesh, iterations=2)

        mesh.visual.vertex_colors = np.concatenate(
            [(col * 255).astype(np.uint8),
             np.full((len(col), 1), 255, dtype=np.uint8)], axis=1
        )
        return mesh

    # ------------------------------------------------------------------ #
    # Node 2 — texture (image+mesh -> smoothed, textured mesh)
    # ------------------------------------------------------------------ #

    def _run_texture(self, params, progress_cb, cancel_event) -> Path:
        import tempfile
        import pymeshlab as ml
        import trimesh

        mesh_path = self._resolve_mesh_path(params)
        smoothing = _SMOOTHING.get(str(params.get("surface_smoothing", "Light")), 10)
        uv_quality = str(params.get("uv_quality", "Trivial"))
        texture_size = int(params.get("texture_size", 2048))
        faces = int(params.get("faces", -1))

        self._report(progress_cb, 5, "Loading mesh…")
        src = trimesh.load(mesh_path, force="mesh")
        if not isinstance(src, trimesh.Trimesh) or len(src.faces) == 0:
            raise ValueError(f"Could not load a valid mesh from: {mesh_path}")

        V = src.vertices.astype(np.float64)
        F = src.faces.astype(np.int32)
        rgba = self._mesh_vertex_rgba(src)

        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(vertex_matrix=V, face_matrix=F, v_color_matrix=rgba), "source")  # mesh 0
        ms.add_mesh(ml.Mesh(vertex_matrix=V, face_matrix=F), "work")                          # mesh 1
        ms.set_current_mesh(1)
        self._check_cancelled(cancel_event)

        # Decimate FIRST: UV unwrap and texture baking cost scale with face count,
        # so bound the working mesh before those steps. Default cap keeps the node
        # responsive even when Generate emitted a very dense mesh. The source mesh
        # (0) stays full density, so baked color stays rich.
        target = faces if faces and faces > 0 else 150000
        if ms.current_mesh().face_number() > target:
            self._report(progress_cb, 20, "Decimating…")
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target))

        if smoothing > 0:
            self._report(progress_cb, 38, "Smoothing surface…")
            ms.apply_coord_taubin_smoothing(stepsmoothnum=smoothing)

        self._report(progress_cb, 55, "UV unwrapping…")
        self._parametrize(ms, uv_quality, texture_size)

        self._report(progress_cb, 70, "Baking texture…")
        ms.transfer_attributes_to_texture_per_vertex(
            sourcemesh=0,
            targetmesh=ms.current_mesh_id(),
            attributeenum="Vertex Color",
            textname="albedo.png",
            textw=texture_size,
            texth=texture_size,
        )
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 90, "Exporting GLB…")
        with tempfile.TemporaryDirectory() as td:
            obj_path = os.path.join(td, "mesh.obj")
            ms.save_current_mesh(obj_path)
            mesh = trimesh.load(obj_path, process=False)
        return self._export(mesh, progress_cb)

    def _parametrize(self, ms, uv_quality: str, texture_size: int) -> None:
        """UV-unwrap the current mesh. Voronoi gives a cleaner atlas but needs a
        manifold mesh, so repair first and fall back to the always-works trivial
        per-triangle atlas."""
        if uv_quality.lower().startswith("voron"):
            try:
                ms.meshing_repair_non_manifold_edges()
            except Exception:
                pass
            try:
                ms.meshing_repair_non_manifold_vertices()
            except Exception:
                pass
            try:
                ms.generate_voronoi_atlas_parametrization()
                return
            except Exception as exc:
                print(f"[TripoSplatGenerator] Voronoi atlas failed ({exc}); using trivial atlas.")
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=int(texture_size))

    def _resolve_mesh_path(self, params: dict) -> str:
        mesh_path = str(params.get("mesh_path", "")).strip()
        if not mesh_path:
            raise ValueError("mesh_path is required for the Texture node (wire a mesh into it).")
        workspace_dir = self.outputs_dir.parent
        p = Path(mesh_path)
        if mesh_path.startswith("/workspace/"):
            return str(workspace_dir / mesh_path[len("/workspace/"):])
        if not p.is_absolute():
            return str(workspace_dir / mesh_path)
        return mesh_path

    @staticmethod
    def _mesh_vertex_rgba(mesh) -> np.ndarray:
        """Per-vertex RGBA in 0..1 from a trimesh, defaulting to mid-grey."""
        n = len(mesh.vertices)
        try:
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) == n:
                rgba = np.asarray(vc, dtype=np.float64) / 255.0
                if rgba.shape[1] == 3:
                    rgba = np.concatenate([rgba, np.ones((n, 1))], axis=1)
                return rgba
        except Exception:
            pass
        return np.tile(np.array([0.7, 0.7, 0.7, 1.0]), (n, 1))

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

    def _gaussian_arrays(self, gaussian):
        """(xyz [N,3], rgb [N,3] in 0..1) from a decoded Gaussian, dropping
        near-transparent points. World coords / activated opacity are exposed via
        get_xyz / get_opacity; ._features_dc holds raw SH DC color."""
        xyz  = gaussian.get_xyz.detach().cpu().float().numpy()
        f_dc = gaussian._features_dc.detach().cpu().float().numpy()  # [N,1,3]
        rgb  = np.clip(f_dc[:, 0, :] * _SH_C0 + 0.5, 0.0, 1.0)
        try:
            opacity = gaussian.get_opacity.detach().cpu().float().numpy().reshape(-1)
            keep = opacity > 0.1
            if keep.sum() > 1000:
                xyz, rgb = xyz[keep], rgb[keep]
        except Exception:
            pass
        return np.ascontiguousarray(xyz, dtype=np.float32), \
               np.ascontiguousarray(rgb, dtype=np.float32)

    @staticmethod
    def _smooth_colors(col: np.ndarray, mesh, iterations: int = 2) -> np.ndarray:
        """Gentle Laplacian smoothing of per-vertex colors over the mesh graph."""
        import scipy.sparse as sp
        edges = mesh.edges_unique
        if len(edges) == 0:
            return col
        n    = len(mesh.vertices)
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        adj  = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
        deg  = np.asarray(adj.sum(1)).ravel()
        deg[deg == 0] = 1.0
        out = col.copy()
        for _ in range(iterations):
            out = 0.5 * out + 0.5 * (adj.dot(out) / deg[:, None])
        return np.clip(out, 0.0, 1.0).astype(np.float32)

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

"""
TripoSplat extension for Modly.

Reference : https://huggingface.co/spaces/VAST-AI/TripoSplat
Weights   : https://huggingface.co/VAST-AI/TripoSplat

TripoSplat is a feed-forward image-to-3D model that outputs 3D Gaussian
Splats (not a mesh). Modly is mesh-centric — every generator returns a .glb and
the viewer only renders GLTF/OBJ. To fit that contract this generator
reconstructs the decoded Gaussians into a watertight, vertex-colored mesh via
screened Poisson surface reconstruction (Open3D) and exports a .glb.

The model code (`triposplat.py` + `model.py`, both pure Python) is taken from
the official Space and bundled in vendor/. Run build_vendor.py once to populate
it; setup.py also fetches it as a fallback at install time.
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

# SH degree-0 -> RGB constant (standard 3D Gaussian Splatting convention)
_SH_C0 = 0.28209479177387814


class TripoSplatGenerator(BaseGenerator):
    MODEL_ID     = "triposplat"
    DISPLAY_NAME = "TripoSplat"
    VRAM_GB      = 10

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        return (self.model_dir / "diffusion_models" / "triposplat_fp16.safetensors").exists()

    def load(self) -> None:
        if self._model is not None:
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
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        steps          = int(params.get("steps", 20))
        guidance_scale = float(params.get("guidance_scale", 3.0))
        num_gaussians  = int(params.get("num_gaussians", 262144))
        mesh_detail    = int(params.get("mesh_detail", 9))
        faces          = int(params.get("faces", -1))
        seed           = int(params.get("seed", -1))
        if seed < 0:
            seed = int(np.random.randint(0, 2**31 - 1))

        pipe  = self._model
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        # 1) Run the model (preprocess + encode + sample + decode in one call).
        #    The per-step callback drives progress and is our cancellation hook
        #    for the long sampling loop.
        self._report(progress_cb, 8, "Generating Gaussians…")

        def _cb(step: int, total: int) -> None:
            if cancel_event and cancel_event.is_set():
                raise GenerationCancelled()
            if progress_cb:
                pct = 8 + int(60 * (step / max(1, total)))
                progress_cb(pct, "Generating Gaussians…")

        gaussian, _prepared = pipe.run(
            image,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            num_gaussians=num_gaussians,
            callback=_cb,
        )
        self._check_cancelled(cancel_event)

        # 2) Gaussians -> mesh
        self._report(progress_cb, 72, "Reconstructing mesh…")
        xyz, rgb = self._gaussian_arrays(gaussian)
        mesh = self._gaussians_to_mesh(xyz, rgb, depth=mesh_detail)
        self._check_cancelled(cancel_event)

        # 3) Optional simplification
        if faces > 0 and len(mesh.faces) > faces:
            self._report(progress_cb, 90, "Simplifying mesh…")
            mesh = self._simplify(mesh, faces)

        # 4) Export
        self._report(progress_cb, 96, "Exporting GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Gaussian -> mesh
    # ------------------------------------------------------------------ #

    def _gaussian_arrays(self, gaussian):
        """Extract (xyz [N,3] float32, rgb [N,3] float32 in 0..1) from a decoded
        Gaussian, dropping near-transparent points to reduce reconstruction noise.

        Note: world-space positions / activated opacity are exposed via the
        get_xyz / get_opacity properties; ._features_dc holds raw SH DC color.
        """
        xyz  = gaussian.get_xyz.detach().cpu().float().numpy()
        f_dc = gaussian._features_dc.detach().cpu().float().numpy()  # [N,1,3]
        rgb  = np.clip(f_dc[:, 0, :] * _SH_C0 + 0.5, 0.0, 1.0)

        try:
            opacity = gaussian.get_opacity.detach().cpu().float().numpy().reshape(-1)
            keep = opacity > 0.1
            if keep.sum() > 1000:  # never let the filter empty the cloud
                xyz, rgb = xyz[keep], rgb[keep]
        except Exception:
            pass

        return np.ascontiguousarray(xyz, dtype=np.float32), \
               np.ascontiguousarray(rgb, dtype=np.float32)

    def _gaussians_to_mesh(self, xyz: np.ndarray, rgb: np.ndarray, depth: int):
        """Screened Poisson reconstruction (Open3D) of a colored point cloud into
        a watertight trimesh. Poisson interpolates the input colors onto the mesh
        vertices, so the result keeps TripoSplat's appearance."""
        import open3d as o3d
        import trimesh

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=16))
        pcd.orient_normals_consistent_tangent_plane(16)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, linear_fit=True,
        )

        # Trim the low-density skirt Poisson adds around the surface.
        densities = np.asarray(densities)
        if densities.size:
            thresh = np.quantile(densities, 0.02)
            mesh.remove_vertices_by_mask(densities < thresh)

        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        if verts.size == 0 or faces.size == 0:
            raise RuntimeError("Poisson reconstruction produced an empty mesh.")

        vcolors = None
        if mesh.has_vertex_colors():
            vc = np.asarray(mesh.vertex_colors, dtype=np.float32)
            vcolors = np.concatenate(
                [np.clip(vc, 0, 1), np.ones((len(vc), 1), dtype=np.float32)], axis=1
            )

        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vcolors, process=False)

    def _simplify(self, mesh, target_faces: int):
        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception as exc:
            print(f"[TripoSplatGenerator] Simplification skipped: {exc}")
            return mesh

    # ------------------------------------------------------------------ #
    # Vendor setup
    # ------------------------------------------------------------------ #

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
                "with the app's Python to build vendor/, or click Repair on the Models "
                "page to re-run setup.py."
            ) from exc

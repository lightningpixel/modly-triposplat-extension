"""
TripoSplat — Modly generator extension (SCAFFOLD ONLY).

Reference: https://github.com/VAST-AI-Research/TripoSplat
           https://huggingface.co/VAST-AI/TripoSplat

This file is a skeleton: it declares the generator class and the methods Modly
requires (`load`, `generate`), but the actual TripoSplat inference is NOT
implemented yet — those methods raise NotImplementedError.

⚠️ Format note: TripoSplat outputs 3D Gaussians (.ply / .splat), NOT a mesh.
Modly's generate() contract expects a path to a .glb mesh, and the viewer /
export pipeline is mesh-centric. Wiring this generator in therefore requires a
product decision (add Gaussian Splatting support to the viewer/export, or
convert Gaussians -> mesh). See the integration ticket. The stubs below leave
that choice open.
"""
import threading
from pathlib import Path
from typing import Callable, Optional

from services.generators.base import BaseGenerator

_HF_REPO_ID = "VAST-AI/TripoSplat"


class TripoSplatGenerator(BaseGenerator):
    MODEL_ID     = "triposplat"
    DISPLAY_NAME = "TripoSplat"
    VRAM_GB      = 8  # estimate — official requirements not published

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        # TODO: refine to check a concrete weight file once the repo layout is
        # confirmed. For now defer to download_check from the manifest.
        return super().is_downloaded()

    def load(self) -> None:
        # TODO: download weights if missing, then load the DINOv3 backbone +
        # diffusion model + VAE decoder onto the right device (cuda / mps / cpu)
        # and assign the loaded pipeline to self._model.
        raise NotImplementedError("TripoSplat load() is not implemented yet.")

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
        # TODO:
        #   1. Preprocess the input image (background removal).
        #   2. Run TripoSplat to produce 3D Gaussians.
        #   3. Decide on the output representation (see module docstring):
        #      either export .glb (requires Gaussian -> mesh conversion) or
        #      extend Modly to handle .ply / .splat outputs.
        #   4. Save to self.outputs_dir and return the resulting path.
        raise NotImplementedError("TripoSplat generate() is not implemented yet.")

    # ------------------------------------------------------------------ #
    # Parameter schema (for the UI)
    # ------------------------------------------------------------------ #

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": -1,
                "min":     -1,
                "max":     4294967295,
                "tooltip": "Seed for reproducibility. Set to -1 for a random seed.",
            },
        ]

# Modly — TripoSplat extension

Single image → 3D using [TripoSplat](https://huggingface.co/spaces/VAST-AI/TripoSplat)
(Tripo AI / VAST AI Research), wired into Modly as a standard image-to-mesh model.

## What it does

TripoSplat is a **feed-forward Gaussian Splatting** model: one forward pass turns
an image into up to 262k 3D Gaussians — fast, no long diffusion-step blowup.

Modly is mesh-centric (the viewer and exporter only handle `.glb` meshes), so this
extension **reconstructs the Gaussians into a watertight, vertex-colored `.glb`**:

```
image → BiRefNet (bg removal) → DINOv3 → flow sampling → Gaussian decode
      → anisotropic density grid (covariance-rasterized) → Surface Nets
      → component cleanup → Taubin smooth → colored .glb mesh
```

The mesh path is **ported from ComfyUI's native `SplatToMesh`** (`splat_mesh.py`):
each Gaussian is rasterized as its oriented 3-sigma covariance disk into a density
grid, the iso-surface is extracted with Surface Nets at an Otsu-picked level, and
vertex colors are sampled from a co-splatted color volume. This respects each
splat's opacity and scale+rotation, so the surface fills solidly instead of
pitting. A mesh is still a lossy approximation of the splats; expect a clean,
recognizable shape rather than the fidelity of a native splat render.

The mesh keeps the Gaussians' color (SH DC → vertex colors sampled from the color volume).

## Parameters

| Param | Default | Notes |
|-------|---------|-------|
| Sampling Steps | 20 | Euler flow-matching steps. 10–25 recommended. |
| Gaussians | 262k | 65k / 131k / 262k. More = denser cloud, finer mesh. |
| Mesh Detail | High (9) | Surface-Nets grid res (7→160 … 10→288). Higher = finer, slower. |
| Mesh Smoothing | Medium | Taubin iterations during reconstruction. Lower = crisper but noisier. |
| Seed | -1 | -1 = random. |

Each run also writes the raw Gaussian splat next to the `.glb` (`.ply` 3DGS + `.splat`)
for viewing in a native splat viewer (e.g. SuperSplat) at full fidelity.

CFG scale, flow shift, and mask erosion are fixed internally (no practical effect
in this image-to-3D path, so kept off the UI).

## Dependencies

Pure PyTorch — **no compiled CUDA extensions**. `setup.py` creates an isolated
venv with an accelerator-matched PyTorch build, plus `numpy safetensors pillow
tqdm huggingface_hub trimesh pymeshlab scipy xatlas`. The model code (`triposplat.py`,
`model.py`) is pure Python and bundled in `vendor/`.

- **Weights:** `VAST-AI/TripoSplat` (~3.8 GB), auto-downloaded on first run.
- **VRAM:** ~10 GB recommended.

## Development

```bash
python build_vendor.py   # refresh vendor/ (triposplat.py + model.py) from the Space
```

Commit `vendor/` so end users never fetch source at runtime.

## License

This extension's own code is released under the [MIT License](LICENSE) © 2026 Lorchie.

### Third-party licenses

The extension downloads and runs third-party models at install/run time, each
under its own license:

| Component | Source | License |
|-----------|--------|---------|
| TripoSplat model & code (`triposplat.py`, `model.py`, weights) | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat) (Tripo AI / VAST AI Research) | MIT |

The MIT license above covers only this extension's code; bundled/downloaded
models remain under the licenses listed here.

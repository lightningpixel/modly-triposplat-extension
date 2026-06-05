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
      → colored point cloud → screened Poisson (Open3D) → .glb mesh
```

The mesh keeps the Gaussians' color (SH DC → vertex colors carried through Poisson).

## Parameters

| Param | Default | Notes |
|-------|---------|-------|
| Sampling Steps | 20 | Euler flow-matching steps. 10–20 recommended. |
| CFG Scale | 3.0 | Guidance strength. ≤1 disables CFG. |
| Gaussians | 262k | 65k / 131k / 262k. More = denser cloud, finer mesh. |
| Mesh Detail | 9 | Poisson depth (7–10). Higher = sharper, slower, more tris. |
| Max Faces | -1 | Quadric decimation target after reconstruction. -1 = off. |
| Seed | -1 | -1 = random. |

## Dependencies

Pure PyTorch — **no compiled CUDA extensions**. `setup.py` creates an isolated
venv with an accelerator-matched PyTorch build, plus `numpy safetensors pillow
tqdm huggingface_hub trimesh open3d`. The model code (`triposplat.py`, `model.py`)
is pure Python and bundled in `vendor/`.

- **Weights:** `VAST-AI/TripoSplat` (~3.8 GB), auto-downloaded on first run.
- **VRAM:** ~10 GB recommended.

## Development

```bash
python build_vendor.py   # refresh vendor/ (triposplat.py + model.py) from the Space
```

Commit `vendor/` so end users never fetch source at runtime.

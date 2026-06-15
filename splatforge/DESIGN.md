# SplatForge — Design Document (Phase 0)

## 1. Integration point — tensor shapes

### What `generator.py` hands in (splat/Z-up frame, BEFORE the `_FLIP`)

| Name | Shape | Dtype | Source |
|---|---|---|---|
| `xyz` | `(N, 3)` | float32, CUDA | `gaussian.get_xyz` |
| `opacity` | `(N,)` | float32, [0,1] | `gaussian.get_opacity` (sigmoid applied) |
| `scale` | `(N, 3)` | float32 (linear std) | `gaussian.get_scaling` (exp applied) |
| `quat` | `(N, 4)` | float32 wxyz | `_rotation + rots_bias` |
| `rgb` | `(N, 3)` | float32, [0,1] | `_features_dc[:,0,:] * _C0 + 0.5` |
| `verts` | `(V, 3)` | float32 | Surface Nets → solidified → decimated (~50k) |
| `faces` | `(F, 3)` | int64 | same |

`N` ≈ 262 144 at "262k (best)" mode. Mesh `V` ≈ 25k vertices / `F` ≈ 50k faces after `_prep_texture_geometry`.

### What splatforge emits

`refine(verts, faces, gaussians, preset, device) → (verts', faces')` — same splat/Z-up frame.
The caller (`_reconstruct_textured` or a new branch) applies the Y/Z flip and the texture bake exactly as today.

---

## 2. Where Σ⁻¹ / opacity / DC color live

### `_inverse_covariance(scale, quat)` — `splat_mesh.py:27`

```python
# Per-splat Σ⁻¹ = R diag(1/s²) Rᵀ
# scale: (N,3) linear std; quat: (N,4) wxyz
# returns: (N,3,3) float32
sinv = _inverse_covariance(scale, quat)
```

`fields.py` must import this directly from `splat_mesh`; do NOT copy/reimplement.

### DC color

```python
_C0 = 0.28209479177387814  # SH band-0 DC → base RGB
rgb = (gaussian._features_dc[:, 0, :].detach().float() * _C0 + 0.5).clamp(0, 1)
```

`_features_dc` shape: `(N, 1, 3)`. TripoSplat outputs DC only — no higher SH bands are used.

### Iso-level τ

`_otsu_level()` in `splat_mesh.py:86` computes the density threshold. The generator does NOT expose `level` externally. `fields.py` must recompute it from the density samples, or the generator must pass it. **Decision for Phase A:** recompute τ from a quick grid sample on a low-res (res=64) density probe rather than re-running `splat_to_mesh`.

---

## 3. Camera conventions (for orbit renders — Phase B)

From `colorize.py` and `generator.py`:

```python
_FLIP = _SPLAT_TO_GLTF = [[1, 0, 0],
                           [0, 0,-1],
                           [0, 1, 0]]   # splat Z-up → glTF Y-up
```

**Splat native frame:** Z-up, Y-forward (depth away from canonical camera).

**Orbit camera setup (Phase B, splat space):**

- **Up axis:** Z = (0, 0, 1)
- **Target:** centroid of the Gaussian cloud ≈ `xyz.mean(0)` (not necessarily the origin)
- **Radius:** `r = 1.5 × max_half_extent`, where `max_half_extent = max(xyz.max(0) - xyz.min(0)) / 2`.
  Typical: r ≈ 1.0–2.0. Must be measured at runtime, NOT hard-coded.
- **Camera positions:** 24–32 directions from icosphere vertices projected onto the sphere of radius r.
- **Intrinsics:** fov ≈ 35°, resolution 512², perspective. Aspect ratio 1:1.
- **extrinsic (lookAt in splat space):**
  ```
  forward = normalize(target - cam_pos)
  right   = normalize(forward × Z_up)      # if forward is near-parallel to Z, fallback right=(1,0,0)
  up_true = right × forward
  R       = [right | up_true | -forward]   # column-major (world-to-camera rotation)
  t       = -R @ cam_pos
  ```
- All rendering in splatforge is done in **splat space**. Apply `_FLIP` only for the final GLB export, outside splatforge.

---

## 4. Analytic density field (fields.py)

```
ρ(x)  = Σᵢ opacity_i · exp(−0.5 · (x−μᵢ)ᵀ Σᵢ⁻¹ (x−μᵢ))
∇ρ(x) = −Σᵢ opacity_i · exp(…) · Σᵢ⁻¹ (x−μᵢ)
```

- Gaussians are **frozen** (μᵢ, Σᵢ⁻¹, opacityᵢ fixed). Only mesh vertex positions x are differentiable.
- Evaluation: cKDTree K=48 nearest Gaussians, chunked at 16 384 query points per batch.
- Σᵢ⁻¹ precomputed once: `sinv = _inverse_covariance(scale.clamp_min(floor), quat)` where `floor = 0.25 × median_spacing` (same floor as `texture_bake._eval_color`).
- Gradients flow through `torch.exp(−0.5 · quad)` w.r.t. x. `quad = einsum("nki,nkij,nkj->nk", d, sinv[idx], d)` where `d = x - centers[idx]`.

---

## 5. Preset wiring plan

### manifest.json addition
```json
{
  "id": "geometry", "label": "Geometry", "type": "select",
  "default": "Standard",
  "options": [
    {"value": "Standard",  "label": "Standard (default)"},
    {"value": "Refined",   "label": "Refined (+90s, best)"},
    {"value": "Maximum",   "label": "Maximum (+5min, experimental)"}
  ]
}
```

### generator.py wiring (after Standard mesh reconstruction)

```python
geometry_preset = str(params.get("geometry", "Standard"))
# Standard: existing path, untouched
# Refined / Maximum: call splatforge.refine() on the raw Surface-Nets verts/faces
if geometry_preset != "Standard":
    from splatforge import refine
    verts, faces = refine(verts, faces, gaussians, preset=geometry_preset, device=dev)
# then texture bake or vertex-color path as today
```

`refine()` receives verts/faces **before** `_prep_texture_geometry` (i.e., the raw solidified surface from `_close_holes` / `_solidify`). It returns improved verts/faces in splat space. Then the existing texture bake / flip chain runs unchanged.

### Phase → preset map

| Preset | Phases | Budget |
|---|---|---|
| Standard | none | ~30s |
| Refined | A + multi-view bake (B.texture) | ≤ 90s |
| Maximum | A + B (full splat renderer) + C | ≤ 5 min |

---

## 6. Architecture checklist (to implement)

```
splatforge/
  __init__.py        → public API: refine(verts, faces, gaussians, preset, device)
  fields.py          → ρ(x), ∇ρ(x), c(x), iso-level recovery
  remesh.py          → edge split / collapse / flip (Botsch-Kobbelt style)
  optimize.py        → Adam loop, losses, schedule
  render/
    splat_render.py  → from-scratch EWA pure-torch renderer
    mesh_raster.py   → z-buffer rasterizer (depth, face_id, barycentrics)
  refine_depth.py    → frozen-visibility Phase C
  bench.py           → PSNR/SSIM, chamfer, timing, VRAM peak
  pipeline.py        → phase A→C orchestration
  DESIGN.md          ← this file
  PROGRESS.md        ← single source of truth
  competition/       ← subagent output logs
```

**Do NOT implement until each STOP gate is approved.**

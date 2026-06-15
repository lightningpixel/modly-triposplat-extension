# SplatForge — Progress

## Current state

**Phase:** A — infrastructure complete. Real bench on 3 objects NOT YET RUN.

### Done
- [x] Phase 0: DESIGN.md written, gate approved
- [x] splatforge/__init__.py — public `refine()` entry point
- [x] splatforge/fields.py — GaussianField (frozen, differentiable ρ, ∇ρ, c), KNN cache
- [x] splatforge/remesh.py — edge split/collapse/flip (Botsch-Kobbelt style, numpy)
- [x] splatforge/optimize.py — competition selector stub (_WINNER = "alpha", pending real bench)
- [x] splatforge/pipeline.py — run_phase_a + refine() orchestration + pre-decimate to 100k
- [x] splatforge/competition/optimize_alpha.py — ALPHA fidelity-first (w_iso=1.0, remesh_every=40)
- [x] splatforge/competition/optimize_beta.py — BETA quality-first (w_iso=0.5, cotangent Lap, remesh_every=20)
- [x] REVIEWER audit: 0 blocking issues on both files
- [x] KNN cache added to fields.py (refresh every 10 calls, force-invalidate after remesh)
- [x] manifest.json: Geometry dropdown (Standard / Refined +90s / Maximum +5min)
- [x] generator.py: geometry_preset wired + splatforge.refine() called + .gaussians.npz saved
- [x] generator.py: 500k ref_pts sampled from filled high-res Surface Nets mesh → saved in .gaussians.npz
- [x] splatforge/bench.py: chamfer_distance() (bidirectional KDTree vs ref_pts) + watertight check in run_bench()
- [x] splatforge/bench.py: edge_cv demoted to informational; min_angle_p5/p50 + aspect_r_p95 as shape tie-breaks
- [x] _bench_phase_a.py: cascade chamfer→iso_fid_delta→min_angle_p5→aspect_r_p95→time; watertight assert; standard-path baseline chamfer; A/B/Standard GLBs saved

### TODO (Phase A gate — real bench not yet run)
- [ ] Restart Modly backend (files synced to installed extension)
- [ ] Run 3 images through Modly with Geometry: Refined, 262k Gaussians — one per category
- [ ] Run bench on each output (commands in Next action below)
- [ ] Report: chamfer table, per-object GLB pairs, VRAM peak
- [ ] Re-adjudicate ALPHA vs BETA; update _WINNER in splatforge/optimize.py
- [ ] Phase A STOP gate review with user; user inspects GLBs before unlocking Phase B

### Locked until Phase A gate
- [ ] Phase B: splat_render.py, mesh_raster.py, multi-view texture
- [ ] Phase C: refine_depth.py (optional)

---

## Last validated gate

Phase 0 — 2026-06-11. User approved with "go".

---

## Next action

1. **Restart Modly backend** (extension files were updated).
2. **Generate 3 objects** — each with Geometry: Refined, Gaussians: 262k Best, any Texture:
   - one organic subject (animal, plant, rounded organic form)
   - one thin-structures subject (chair, bicycle, fence, plant with stems)
   - one hard-surface subject (vehicle, tool, architectural element)
3. **Run bench for each** (from the dev dir `modly-triposplat-extension/`):

```
python _bench_phase_a.py 200 --obj organic --npz "PATH_TO_OUTPUT.gaussians.npz"
python _bench_phase_a.py 200 --obj thin    --npz "PATH_TO_OUTPUT.gaussians.npz"
python _bench_phase_a.py 200 --obj hard    --npz "PATH_TO_OUTPUT.gaussians.npz"
```

The `.gaussians.npz` is saved next to each output `.glb` in the Modly workspace.
Output GLBs land in `splatforge/competition/bench_out/` for visual comparison.

4. **Report back**: paste the `=== COMPETITION ===` block for each object + confirm watertight gate passed.

---

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-06-11 | refine() receives verts/faces BEFORE _prep_texture_geometry | Optimizer should run on raw solid surface, not 50k cap |
| 2026-06-11 | Iso-level tau recomputed in GaussianField via res=64 grid probe | Avoids re-running splat_to_mesh; generator doesn't expose level |
| 2026-06-11 | Camera radius = 1.5 × max_half_extent (measured at runtime) | Object scale not fixed; hard-coding breaks some models |
| 2026-06-11 | KNN batch-computed for all V vertices once per _KNN_REFRESH calls | Per-chunk KNN (old design) was ~2s/step; batch+cache → ~0.2s/step |
| 2026-06-11 | Competition: main session writes fields/remesh/bench, agents write optimize variants | fields.py math must be identical; only loss weights / Laplacian type differ |
| 2026-06-11 | REVIEWER (Opus): 0 blocking issues on both ALPHA and BETA | Both pass VRAM/NaN/license/budget/API checks |
| 2026-06-11 | pipeline.refine() pre-decimates to 100k before optimizer | Raw post-fill surface can be 800k+ faces; 100k is within VRAM/time budget |
| 2026-06-11 | edge_cv Δ = after - before (standard convention) | Prior bench used before-after (non-standard, confusing sign); fixed in bench.py |
| 2026-06-11 | Replaced edge_cv as triangle-quality tie-break with min_angle_p5/p50 + aspect_r_p95 | edge_cv penalizes adaptive remeshing by construction (intentionally non-uniform edge lengths); min-angle and aspect-ratio are shape-only, scale-invariant — fair for adaptive meshes |
| 2026-06-11 | edge_cv kept as informational metric only (not used in winner selection) | Preserves historical data; documents the adaptive-sizing confound |
| 2026-06-11 | iso_fid demoted from #1 to #2; chamfer added as #1 | iso_fid is the optimizer's own training loss — circular, rewards overfitting, blind to tangential sliding. Chamfer vs high-res ref is independent of the loss |
| 2026-06-11 | Adjudication order: chamfer > iso_fid_delta > min_angle_p5 > aspect_r_p95 > time | Primary = independent fidelity; secondary = optimizer convergence; tie-breaks = shape quality; final = speed |
| 2026-06-11 | ref_pts (500k surface samples from filled high-res Surface Nets mesh) saved in .gaussians.npz | Reference pre-dates all decimation; used for chamfer; independent of optimizer |
| 2026-06-11 | Watertight preservation added as spec gate in bench (assert fires after all metrics printed) | Spec requirement: optimizer must not break topology of input mesh |

---

## Known issues / parked

- xatlas segfault on Windows above ~120k faces — not a splatforge issue (handled upstream).
- _mean_edge_length_np only samples (0,1) edge per face — cosmetic, proxy is fine.
- Beta's cotangent NaN fallback is belt-and-suspenders (source clamped already) — harmless.
- For Refined+Baked: _prep_texture_geometry's volumetric resample partially undoes refinement.
  Acceptable for Phase A gate; will skip or replace resample step post-gate.

---

## edge_cv semantics (clarification for proxy bench result)

Proxy bench (30 steps, constant opacity/scale/quat):
- Both iso_fid tied: 0.6040 → 0.3591 (iso_fid_delta = -0.2449)
- ALPHA edge_cv: 0.418 → 0.452 (edge_cv_delta = +0.034, worsened by 0.034)
- BETA  edge_cv: 0.418 → 0.470 (edge_cv_delta = +0.052, worsened by 0.052)
- Both degraded triangle uniformity (expected — iso pull moves verts off uniform grid)
- ALPHA degraded LESS → ALPHA wins on triangle quality (proxy bench; tie on iso_fid)
- Winner set to ALPHA as tie-break on edge_cv + speed (1.7× faster)
- INCONCLUSIVE: proxy bench cannot rank ALPHA vs BETA; real bench required

---

## Bench history

| Date | Object | Preset | n_steps | Gaussians | iso_fid_delta ALPHA | iso_fid_delta BETA | edge_cv_delta ALPHA | edge_cv_delta BETA | t_ALPHA | t_BETA | Winner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-11 | proxy | Refined | 30 | fake (const) | -0.2449 | -0.2449 | +0.034 | +0.052 | 2.9s | 5.0s | ALPHA (tie-break) |

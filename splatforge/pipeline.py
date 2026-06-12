"""
Orchestration: Phase A → (B, C pending) per preset.
Returns (verts, faces) in splat/Z-up frame; caller handles flip + texture bake.

STATUS (2026-06-11): Phase A is CLOSED — not exposed in the manifest UI.
Bench on 3 valid self-contained inputs showed the iso-surface optimizer cannot
improve chamfer vs the Standard extraction: the reference surface is a Taubin-
smoothed grid extraction (not an iso-surface of the analytic field), and the
Standard mesh is already its direct resample, i.e. at the noise floor.
The aligned GaussianField (0.9×spacing floor, surface-aligned tau), the bench
harness and the self-contained npz contract are kept for Phase B: SuGaR-style
joint refinement through differentiable splat rendering, gated on PSNR/SSIM
over held-out views (independent criterion, no parent-mesh circularity).
"""
import time
import numpy as np
import torch

# Hard time budgets per preset (seconds); leaves headroom for texture bake.
_BUDGETS      = {"Refined": 75.0, "Maximum": 270.0}
_N_STEPS      = {"Refined": 200,  "Maximum": 400}
_LR           = {"Refined": 5e-4, "Maximum": 1e-3}
_OPT_FACE_CAP = 100_000  # working budget fed to the Phase A optimizer


def _decimate(verts: np.ndarray, faces: np.ndarray, target: int):
    import pymeshlab as ml
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=verts.astype(np.float64), face_matrix=faces.astype(np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), preservetopology=True)
    cm = ms.current_mesh()
    return (np.ascontiguousarray(cm.vertex_matrix(), np.float32),
            np.ascontiguousarray(cm.face_matrix(), np.int64))


def run_phase_a(verts: np.ndarray, faces: np.ndarray,
                gaussians: dict, preset: str,
                device: torch.device) -> tuple:
    """Phase A: field-space geometry optimizer. Returns (verts, faces)."""
    from .fields import GaussianField
    from .optimize import optimize_phase_a, get_default_weights
    from .bench import run_bench

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"[splatforge] Building GaussianField …", flush=True)
    field = GaussianField(gaussians, device, surface_pts=verts)  # tau aligned on input surface
    print(f"[splatforge] τ = {field.tau:.4f} (surface-aligned)  ({time.time()-t0:.1f}s)", flush=True)

    budget  = _BUDGETS.get(preset, 75.0) - (time.time() - t0)
    n_steps = _N_STEPS.get(preset, 200)
    lr      = _LR.get(preset, 5e-4)
    weights = get_default_weights(preset)

    print(f"[splatforge] Phase A: {n_steps} steps  lr={lr}  budget={budget:.0f}s", flush=True)
    verts_out, faces_out = optimize_phase_a(
        verts, faces, field, n_steps=n_steps, lr=lr,
        weights=weights, device=device, budget_s=budget)

    run_bench(verts, faces, verts_out, faces_out, field,
              tag=preset, t_elapsed=time.time() - t0)
    return verts_out, faces_out


def refine(verts: np.ndarray, faces: np.ndarray,
           gaussians: dict, preset: str,
           device: torch.device) -> tuple:
    """Public API: refine(verts, faces, gaussians, preset, device) -> (verts, faces).

    verts/faces in splat/Z-up frame; returned in the same frame.
    Presets: "Refined" (<=90s end-to-end) | "Maximum" (<=5min).
    Input mesh may be the raw post-fill surface (800k+ faces); decimated internally
    to _OPT_FACE_CAP before the optimizer to keep VRAM and time within budget.
    """
    if preset not in ("Refined", "Maximum"):
        raise ValueError(f"[splatforge] Unknown preset {preset!r}")

    if len(faces) > _OPT_FACE_CAP:
        print(f"[splatforge] Decimating {len(faces)} -> {_OPT_FACE_CAP} faces for optimizer …", flush=True)
        verts, faces = _decimate(verts, faces, _OPT_FACE_CAP)

    verts_out, faces_out = run_phase_a(verts, faces, gaussians, preset, device)

    # Phase B (multi-view texture) and Phase C (depth refinement) are placeholders
    # until their STOP gates are approved.

    return verts_out, faces_out

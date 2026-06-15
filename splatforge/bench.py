"""
Phase A metrics: iso-surface fidelity, chamfer distance, and triangle quality.
Phase B adds per-view PSNR/SSIM once the splat renderer exists.

Adjudication cascade (ALPHA vs BETA):
  1. chamfer         lower wins  -- independent fidelity vs high-res ref mesh
  2. iso_fid_delta   more negative wins  -- secondary (training loss, can overfit)
  3. min_angle_p5    higher wins  -- shape quality, scale-invariant
  4. aspect_r_p95    lower wins   -- shape quality, scale-invariant
  5. time            lower wins

Triangle-quality metrics (scale-invariant, shape-only):
  min_angle_p5/p50  -- 5th/50th-percentile of per-face minimum interior angle (degrees).
                       Higher = more equilateral. 60 deg = ideal equilateral triangle.
  aspect_r_p95      -- 95th-percentile of longest/shortest edge per face. 1.0 = equilateral.
  edge_cv           -- coefficient of variation of all edge lengths.
                       INFORMATIONAL ONLY: penalizes adaptive remeshing by construction.
"""
import numpy as np
import torch


# ── geometry helpers ──────────────────────────────────────────────────────────

def _face_angles(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(F, 3) interior angles in radians."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    def _ang(a, b):
        n = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        return np.arccos(np.clip(np.einsum("ij,ij->i", a, b) / n, -1.0, 1.0))
    a0 = _ang(v1 - v0, v2 - v0)
    a1 = _ang(v0 - v1, v2 - v1)
    a2 = np.clip(np.pi - a0 - a1, 0.0, np.pi)
    return np.stack([a0, a1, a2], axis=1)


def _aspect_ratios(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face aspect ratio = longest_edge / shortest_edge. Equilateral = 1.0."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    edges = np.stack([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1),
    ], axis=1)
    return edges.max(axis=1) / (edges.min(axis=1) + 1e-12)


# ── public metric functions ───────────────────────────────────────────────────

def iso_fidelity(verts: np.ndarray, field) -> float:
    """Mean squared deviation of vertex density from the iso-level: E[(rho(v)-tau)^2]."""
    v_t = torch.as_tensor(verts, device=field.device)
    with torch.no_grad():
        rho = field.density(v_t).cpu().numpy()
    return float(((rho - field.tau) ** 2).mean())


def chamfer_distance(ref_pts: np.ndarray, verts: np.ndarray, faces: np.ndarray,
                     n_sample_cand: int = 100_000) -> dict:
    """Bidirectional chamfer distance between ref_pts and the candidate mesh surface.

    Metric: mean L1 (average Euclidean nearest-neighbour distance) — NOT squared.
      cd_mean = 0.5 * (mean_ref2cand + mean_cand2ref)

    ref_pts       : (N, 3) float32 — 500 k points surface-sampled (trimesh uniform) from
                    the filled high-res Surface Nets mesh (224-grid resolution) in
                    splat/Z-up frame, saved to .gaussians.npz before the Y-up flip.
                    IMPORTANT: must be in the same coordinate frame as verts/faces.
                    If you pass Y-up verts against Z-up ref_pts (e.g. loading a GLB that
                    was exported post-_FLIP without un-flipping first), the distance will
                    be on the order of sqrt(2) × RMS(y,z) — typically 0.3–0.8 × diag for
                    asymmetric objects. Un-flip first: verts_zup = verts_yup @ _FLIP_INV.
    verts / faces : candidate mesh — must be in the SAME frame as ref_pts (Z-up after the
                    un-flip applied at bench load time).
    n_sample_cand : surface samples drawn from the candidate for the cand→ref direction.

    Returns {"ref2cand": float, "cand2ref": float, "mean": float}.
    Reading both directions separately reveals the failure mode:
      ref2cand high → candidate misses reference coverage (under-coverage / holes)
      cand2ref high → candidate has extra geometry absent from reference (floaters/inflation)
      both equal    → uniform offset, consistent with a resolution gap between the
                       224-res reference and the coarser (160-res) resampled candidate.
                       At 160 cells across diag ≈ 1.38, one voxel ≈ 0.0086 units.
                       A cd_mean ≈ 0.01 (1–2 voxels) is expected from resolution alone;
                       0.05–0.1 indicates thin-feature loss (legs, ears) or frame drift.
    """
    import trimesh
    from scipy.spatial import cKDTree

    cand_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    cand_pts, _ = trimesh.sample.sample_surface(cand_mesh, n_sample_cand)
    cand_pts = np.asarray(cand_pts, np.float32)

    d_ref2cand = float(cKDTree(cand_pts).query(ref_pts,  k=1, workers=-1)[0].mean())
    d_cand2ref = float(cKDTree(ref_pts).query(cand_pts, k=1, workers=-1)[0].mean())
    return {"ref2cand": d_ref2cand, "cand2ref": d_cand2ref, "mean": (d_ref2cand + d_cand2ref) / 2}


def mesh_quality(verts: np.ndarray, faces: np.ndarray) -> dict:
    """Scale-invariant per-face triangle quality + informational edge stats."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    elen = np.concatenate([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1),
    ])
    min_ang_deg = np.degrees(_face_angles(verts, faces).min(axis=1))
    ar = _aspect_ratios(verts, faces)
    return {
        "min_angle_p5":  float(np.percentile(min_ang_deg, 5)),
        "min_angle_p50": float(np.percentile(min_ang_deg, 50)),
        "aspect_r_p95":  float(np.percentile(ar, 95)),
        "edge_mean":     float(elen.mean()),
        "edge_cv":       float(elen.std() / (elen.mean() + 1e-8)),
        "n_faces":       len(faces),
        "n_verts":       len(verts),
    }


def vram_peak_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


# ── main bench entry point ────────────────────────────────────────────────────

def run_bench(verts_before, faces_before, verts_after, faces_after,
              field, tag: str = "", t_elapsed: float = 0.0,
              ref_pts: np.ndarray = None) -> dict:
    """Print + return Phase A metrics comparing before/after geometry.

    Delta convention: after - before.
      iso_fid_delta < 0  -> improved (closer to iso-surface)
      chamfer            -> lower = better independent fidelity (vs high-res ref)
      min_angle_p5 after -> higher = more equilateral worst-case faces
      aspect_r_p95 after -> lower = less degenerate worst-case faces
      edge_cv            -> informational only (not in adjudication cascade)

    Watertight:
      wt_preserved = True iff input was not watertight, OR output is still watertight.
      Gate requirement: wt_preserved must hold. Reported but not asserted here
      (caller asserts so all objects complete before abort).
    """
    import trimesh

    fid_b = iso_fidelity(verts_before, field)
    fid_a = iso_fidelity(verts_after,  field)
    qb    = mesh_quality(verts_before, faces_before)
    qa    = mesh_quality(verts_after,  faces_after)
    vram  = vram_peak_gb()

    wt_before = trimesh.Trimesh(vertices=verts_before, faces=faces_before, process=False).is_watertight
    wt_after  = trimesh.Trimesh(vertices=verts_after,  faces=faces_after,  process=False).is_watertight
    wt_preserved = (not wt_before) or wt_after

    _nan = float("nan")
    if ref_pts is not None:
        _cd = chamfer_distance(ref_pts, verts_after, faces_after)
        cd_ref2cand = _cd["ref2cand"]
        cd_cand2ref = _cd["cand2ref"]
        cd_mean     = _cd["mean"]
    else:
        cd_ref2cand = cd_cand2ref = cd_mean = _nan

    result = {
        "tag":              tag,
        # fidelity — both chamfer directions + mean for adjudication
        "chamfer":          cd_mean,
        "chamfer_ref2cand": cd_ref2cand,
        "chamfer_cand2ref": cd_cand2ref,
        "iso_fid_before":   fid_b,
        "iso_fid_after":    fid_a,
        "iso_fid_delta":    fid_a - fid_b,
        # triangle quality — tie-break metrics
        "min_angle_p5_before":  qb["min_angle_p5"],
        "min_angle_p5_after":   qa["min_angle_p5"],
        "min_angle_p50_before": qb["min_angle_p50"],
        "min_angle_p50_after":  qa["min_angle_p50"],
        "aspect_r_p95_before":  qb["aspect_r_p95"],
        "aspect_r_p95_after":   qa["aspect_r_p95"],
        # watertight gate
        "wt_before":    wt_before,
        "wt_after":     wt_after,
        "wt_preserved": wt_preserved,
        # informational
        "edge_cv_before":  qb["edge_cv"],
        "edge_cv_after":   qa["edge_cv"],
        "edge_cv_delta":   qa["edge_cv"] - qb["edge_cv"],
        "n_faces_before":  qb["n_faces"],
        "n_faces_after":   qa["n_faces"],
        "t_s":             t_elapsed,
        "vram_gb":         vram,
    }

    def _fmt(v): return f"{v:.6f}" if not np.isnan(v) else "n/a"
    _wt_str  = ("OK" if wt_preserved else "FAIL wt lost") + f"  ({wt_before}->{wt_after})"
    print(
        f"[bench {tag}]\n"
        f"  chamfer   mean={_fmt(cd_mean)}  ref->cand={_fmt(cd_ref2cand)}  cand->ref={_fmt(cd_cand2ref)}\n"
        f"  iso_fid   {fid_b:.4f} -> {fid_a:.4f}  (d{result['iso_fid_delta']:+.4f})\n"
        f"  min_angle p5: {qb['min_angle_p5']:.1f} -> {qa['min_angle_p5']:.1f} deg"
        f"    p50: {qb['min_angle_p50']:.1f} -> {qa['min_angle_p50']:.1f} deg\n"
        f"  aspect_r  p95: {qb['aspect_r_p95']:.2f} -> {qa['aspect_r_p95']:.2f}\n"
        f"  edge_cv   {qb['edge_cv']:.3f} -> {qa['edge_cv']:.3f}"
        f"  (d{result['edge_cv_delta']:+.3f}, info only)\n"
        f"  watertight: {_wt_str}   "
        f"faces {qb['n_faces']} -> {qa['n_faces']}"
        f"   {t_elapsed:.1f}s  {vram:.2f}GB",
        flush=True,
    )
    return result

"""
Phase A bench — Standard baseline + ALPHA + HYBRID on real TripoSplat Gaussians.

Usage:
  python _bench_phase_a.py [N_STEPS] [--obj LABEL] [--npz PATH] [--seeds N]

  N_STEPS      optimizer steps (default 200; use 50 for smoke-test)
  --obj LABEL  object label for output filenames (organic / table / voiture)
  --npz PATH   explicit .gaussians.npz; if omitted uses latest in INSTALLED/_full_test_out/
  --seeds N    chamfer measurement seeds per variant (default 3)

Gate criterion:
  HYBRID cd_mean < Standard cd_mean on THIS object (absolute, not relative to ALPHA).
  Budget: HYBRID must finish within 90 s on the worst (table) object.

Input contract (self-contained npz — no GLB pairing):
  The npz MUST embed start_verts/start_faces (Standard mesh, Z-up, already prepped
  at 160-res + 50k faces by generator._save_splats) alongside gaussians + ref_pts.
  Everything lives in the splat/Z-up frame; output GLBs apply _FLIP for the viewer.
  Start mesh and ref_pts are HARD-validated against the gaussian cloud bbox/centroid;
  any mismatch aborts the bench (a gate harness refuses invalid input).

Outputs:
  splatforge/competition/bench_out/{obj}_standard.glb
  splatforge/competition/bench_out/{obj}_ALPHA_{N_STEPS}steps.glb
  splatforge/competition/bench_out/{obj}_HYBRID_{N_STEPS}steps.glb
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, torch, trimesh

# ── paths ─────────────────────────────────────────────────────────────────────
EXT       = Path(__file__).parent
INSTALLED = Path(r"C:\Users\LORCHIE\Documents\Modly\extensions\triposplat")
sys.path.insert(0, str(INSTALLED))  # vendor/triposplat modules
sys.path.insert(0, str(EXT))        # dev splatforge overrides installed copy

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("n_steps", nargs="?", type=int, default=200)
parser.add_argument("--obj",   default="obj",  help="Object label for filenames")
parser.add_argument("--npz",   default="",     help="Explicit .gaussians.npz path")
parser.add_argument("--seeds", type=int, default=3, help="Chamfer measurement seeds")
args    = parser.parse_args()
N_STEPS = args.n_steps
OBJ     = args.obj
N_SEEDS = args.seeds
dev     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# _FLIP: Z-up → Y-up, applied only at GLB export (all bench geometry stays Z-up).
_FLIP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)

# ── Load Gaussians + ref_pts ───────────────────────────────────────────────────
if args.npz:
    npz_path = Path(args.npz)
    assert npz_path.exists(), f"npz not found: {npz_path}"
else:
    candidates = sorted((INSTALLED / "_full_test_out").glob("*.gaussians.npz"))
    if not candidates:
        candidates = sorted(INSTALLED.rglob("*.gaussians.npz"))
    assert candidates, (
        "No .gaussians.npz found. Run the extension with any image first "
        "(it saves one next to every GLB), or pass --npz PATH."
    )
    npz_path = candidates[-1]

print(f"Gaussians npz: {npz_path}", flush=True)
data = np.load(str(npz_path))

_GAUSS_KEYS = {"xyz", "opacity", "scale", "quat", "rgb"}
gaussians = {k: torch.as_tensor(data[k], dtype=torch.float32).to(dev)
             for k in data.files if k in _GAUSS_KEYS}
# Same min_opacity filter as splat_to_mesh (the reference surface ignores these).
_keep = gaussians["opacity"].reshape(-1) >= 0.02
gaussians = {k: v[_keep] for k, v in gaussians.items()}
n = len(gaussians["xyz"])

# ref_pts are in Z-up (sampled before _FLIP in generator._reconstruct)
ref_pts = data["ref_pts"].astype(np.float32) if "ref_pts" in data.files else None
if ref_pts is not None:
    print(f"  N={n}  ref_pts={len(ref_pts)}", flush=True)
else:
    print(f"  N={n}  ref_pts=MISSING (no chamfer — re-generate with updated extension)",
          flush=True)

# ── Load embedded start mesh (self-contained npz — no GLB pairing) ───────────
# start_verts/start_faces are embedded by generator._save_splats: the Standard
# mesh (Z-up, pre-refine) already prepped (160-res resample + 50k decimation).
assert "start_verts" in data.files and "start_faces" in data.files, (
    f"{npz_path.name} has no embedded start mesh (old format). "
    "Re-generate the object in Modly with the updated extension — the bench "
    "refuses filename-based GLB pairing (mispair risk)."
)
verts = np.ascontiguousarray(data["start_verts"], np.float32)
faces = np.ascontiguousarray(data["start_faces"], np.int64)
print(f"  Start mesh (embedded, Z-up): {len(faces)} faces  {len(verts)} verts", flush=True)

# ── HARD input validation: start mesh must match the gaussian cloud ──────────
# A gate harness refuses invalid input; it never warns-and-proceeds.
g_xyz  = gaussians["xyz"].cpu().numpy()
g_lo, g_hi = g_xyz.min(0), g_xyz.max(0)
m_lo, m_hi = verts.min(0), verts.max(0)
diag   = float(np.linalg.norm(g_hi - g_lo))
c_gap  = float(np.linalg.norm(verts.mean(0) - g_xyz.mean(0)))
bb_gap = float(max(np.abs(m_lo - g_lo).max(), np.abs(m_hi - g_hi).max()))
_TOL_CENTROID = 0.10   # fraction of gaussian-cloud diag
_TOL_BBOX     = 0.15
if c_gap > _TOL_CENTROID * diag or bb_gap > _TOL_BBOX * diag:
    raise SystemExit(
        f"INPUT VALIDATION FAIL: start mesh does not match gaussian cloud.\n"
        f"  centroid gap = {c_gap:.4f} ({c_gap/diag*100:.1f}% of diag, tol {_TOL_CENTROID*100:.0f}%)\n"
        f"  bbox gap     = {bb_gap:.4f} ({bb_gap/diag*100:.1f}% of diag, tol {_TOL_BBOX*100:.0f}%)\n"
        f"  npz: {npz_path}")
print(f"  Input validation OK: centroid gap {c_gap/diag*100:.1f}%  bbox gap {bb_gap/diag*100:.1f}%",
      flush=True)

if ref_pts is not None:
    c_ref = float(np.linalg.norm(verts.mean(0) - ref_pts.mean(0)))
    if c_ref > _TOL_CENTROID * diag:
        raise SystemExit(
            f"INPUT VALIDATION FAIL: ref_pts do not match start mesh "
            f"(centroid gap {c_ref/diag*100:.1f}% of diag). npz: {npz_path}")

# ── Chamfer helper — 3-seed mean ± std ────────────────────────────────────────
from splatforge.bench import chamfer_distance, run_bench

def _chamfer_3x(ref, v, f, seeds):
    """Surface-sample chamfer N_SEEDS times, return mean±std dict."""
    runs = []
    for s in seeds:
        np.random.seed(s)
        runs.append(chamfer_distance(ref, v, f))
    means   = [r["mean"]     for r in runs]
    r2cs    = [r["ref2cand"] for r in runs]
    c2rs    = [r["cand2ref"] for r in runs]
    return {
        "mean":       float(np.mean(means)),
        "std":        float(np.std(means, ddof=0)),
        "ref2cand":   float(np.mean(r2cs)),
        "cand2ref":   float(np.mean(c2rs)),
    }

CD_SEEDS = list(range(N_SEEDS))

# ── Standard baseline chamfer ─────────────────────────────────────────────────
cd_std = {"mean": float("nan"), "std": float("nan"),
           "ref2cand": float("nan"), "cand2ref": float("nan")}
if ref_pts is not None:
    t_cd = time.time()
    cd_std = _chamfer_3x(ref_pts, verts, faces, CD_SEEDS)
    diag   = float(np.linalg.norm(np.array(verts).max(0) - np.array(verts).min(0)))
    print(f"Standard chamfer  mean={cd_std['mean']:.6f}±{cd_std['std']:.6f}"
          f"  r->c={cd_std['ref2cand']:.6f}  c->r={cd_std['cand2ref']:.6f}"
          f"  (diag={diag:.3f}, cd/diag={cd_std['mean']/diag*100:.1f}%)"
          f"  ({time.time()-t_cd:.1f}s)", flush=True)

# ── GaussianField ─────────────────────────────────────────────────────────────
from splatforge.fields import GaussianField

t0    = time.time()
field = GaussianField(gaussians, dev, surface_pts=verts)  # tau aligned on start surface
print(f"GaussianField built in {time.time()-t0:.1f}s  tau={field.tau:.4f} (surface-aligned)",
      flush=True)

# ── Output dir ────────────────────────────────────────────────────────────────
OUT = EXT / "splatforge" / "competition" / "bench_out"
OUT.mkdir(parents=True, exist_ok=True)

# Export standard GLB (verts Z-up → apply _FLIP for Y-up viewer)
trimesh.Trimesh(vertices=(verts @ _FLIP).astype(np.float32),
                faces=faces.astype(np.int64), process=False
                ).export(str(OUT / f"{OBJ}_standard.glb"))

# ── ALPHA ─────────────────────────────────────────────────────────────────────
from splatforge.competition.optimize_alpha import optimize_phase_a as alpha_fn, DEFAULT_WEIGHTS as alpha_w

field.refresh_knn_cache()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t_a  = time.time()
va, fa = alpha_fn(verts.copy(), faces.copy(), field,
                  n_steps=N_STEPS, lr=5e-4,
                  weights=alpha_w["Refined"], device=dev, budget_s=300.0)
t_a  = time.time() - t_a
field.refresh_knn_cache()
r_a  = run_bench(verts, faces, va, fa, field,
                 tag=f"ALPHA/{OBJ}", t_elapsed=t_a, ref_pts=None)  # chamfer measured below
cd_a = _chamfer_3x(ref_pts, va, fa, CD_SEEDS) if ref_pts is not None else cd_std.copy()
trimesh.Trimesh(vertices=(va @ _FLIP).astype(np.float32),
                faces=fa.astype(np.int64), process=False
                ).export(str(OUT / f"{OBJ}_ALPHA_{N_STEPS}steps.glb"))

# ── HYBRID ────────────────────────────────────────────────────────────────────
from splatforge.competition.optimize_hybrid import (
    optimize_phase_a as hybrid_fn, DEFAULT_WEIGHTS as hybrid_w)

field.refresh_knn_cache()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t_h  = time.time()
vh, fh = hybrid_fn(verts.copy(), faces.copy(), field,
                   n_steps=N_STEPS, lr=5e-4,
                   weights=hybrid_w["Refined"], device=dev, budget_s=90.0,
                   profile=True)
t_h  = time.time() - t_h
field.refresh_knn_cache()
r_h  = run_bench(verts, faces, vh, fh, field,
                 tag=f"HYBRID/{OBJ}", t_elapsed=t_h, ref_pts=None)
cd_h = _chamfer_3x(ref_pts, vh, fh, CD_SEEDS) if ref_pts is not None else cd_std.copy()
trimesh.Trimesh(vertices=(vh @ _FLIP).astype(np.float32),
                faces=fh.astype(np.int64), process=False
                ).export(str(OUT / f"{OBJ}_HYBRID_{N_STEPS}steps.glb"))

# ── Watertight gate ───────────────────────────────────────────────────────────
_wt_in = trimesh.Trimesh(vertices=verts, faces=faces, process=False).is_watertight
print(f"\n[watertight gate]  input={_wt_in}"
      f"  ALPHA={r_a['wt_after']}(ok={r_a['wt_preserved']})"
      f"  HYBRID={r_h['wt_after']}(ok={r_h['wt_preserved']})", flush=True)
_wt_failures = []
if not r_a["wt_preserved"]:
    _wt_failures.append("ALPHA broke watertight")
if not r_h["wt_preserved"]:
    _wt_failures.append("HYBRID broke watertight")

# ── Gate verdict ──────────────────────────────────────────────────────────────
_nan = float("nan")
def _gt(v): return f"{v:.6f}" if not np.isnan(v) else "n/a"
def _gt_pm(m, s): return (f"{m:.6f}±{s:.6f}" if not np.isnan(m) else "n/a")

print(f"\n=== BENCH [{OBJ}] {N_STEPS} steps  ({N_SEEDS} chamfer seeds) ===", flush=True)
hdr = (f"{'variant':<10}  {'cd_mean±std':>18}  {'r->c':>10}  {'c->r':>10}"
       f"  {'iso_fid_d':>10}  {'minang_p5':>9}  {'ar_p95':>6}"
       f"  {'edgecv_d':>9}(info)  {'time':>6}  {'VRAM':>5}")
print(hdr)

std_row = (f"{'standard':<10}  {_gt_pm(cd_std['mean'],cd_std['std']):>18}"
           f"  {_gt(cd_std['ref2cand']):>10}  {_gt(cd_std['cand2ref']):>10}"
           f"  {'n/a':>10}  {'n/a':>9}  {'n/a':>6}  {'n/a':>9}        {'n/a':>6}  {'n/a':>5}")
alp_row = (f"{'ALPHA':<10}  {_gt_pm(cd_a['mean'],cd_a['std']):>18}"
           f"  {_gt(cd_a['ref2cand']):>10}  {_gt(cd_a['cand2ref']):>10}"
           f"  {r_a['iso_fid_delta']:>+10.4f}  {r_a['min_angle_p5_after']:>9.1f}d"
           f"  {r_a['aspect_r_p95_after']:>6.2f}  {r_a['edge_cv_delta']:>+9.3f}"
           f"        {t_a:>6.1f}s  {r_a['vram_gb']:>5.2f}GB")
hyb_row = (f"{'HYBRID':<10}  {_gt_pm(cd_h['mean'],cd_h['std']):>18}"
           f"  {_gt(cd_h['ref2cand']):>10}  {_gt(cd_h['cand2ref']):>10}"
           f"  {r_h['iso_fid_delta']:>+10.4f}  {r_h['min_angle_p5_after']:>9.1f}d"
           f"  {r_h['aspect_r_p95_after']:>6.2f}  {r_h['edge_cv_delta']:>+9.3f}"
           f"        {t_h:>6.1f}s  {r_h['vram_gb']:>5.2f}GB")
print(std_row)
print(alp_row)
print(hyb_row)

# Gate check (must beat Standard)
eps = 0.0
gate_alpha  = (not np.isnan(cd_a["mean"])) and cd_a["mean"] < cd_std["mean"] - eps
gate_hybrid = (not np.isnan(cd_h["mean"])) and cd_h["mean"] < cd_std["mean"] - eps
gate_budget = t_h <= 90.0

print(f"\n[gate]  ALPHA < Standard: {gate_alpha} ({cd_a['mean']:.6f} vs {cd_std['mean']:.6f})")
print(f"[gate]  HYBRID < Standard: {gate_hybrid} ({cd_h['mean']:.6f} vs {cd_std['mean']:.6f})")
print(f"[gate]  HYBRID budget <=90s: {gate_budget} ({t_h:.1f}s)")

if gate_hybrid and gate_budget:
    winner = "HYBRID"
elif gate_alpha:
    winner = "ALPHA (HYBRID failed gate)"
else:
    winner = "NONE — Phase A gate FAILS (neither beats Standard)"

print(f"\n>>> [{OBJ}] Winner: {winner}")
print(f"GLBs: {OUT}/")
print(f"  {OBJ}_standard.glb")
print(f"  {OBJ}_ALPHA_{N_STEPS}steps.glb")
print(f"  {OBJ}_HYBRID_{N_STEPS}steps.glb")

# Spec gate asserts — fire AFTER full table is printed
assert not _wt_failures, f"SPEC GATE FAIL: {'; '.join(_wt_failures)}"

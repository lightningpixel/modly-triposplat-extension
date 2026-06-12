"""
Held-out camera rig for Phase B evaluation.

K viewpoints on a sphere around the object (Fibonacci lattice — deterministic,
no seed needed, near-uniform angular coverage). All geometry is in the splat/
Z-up frame. Cameras look at the bbox center; OpenCV-style pinhole convention:
camera looks down +Z, x right, y DOWN. Returns world-to-camera extrinsics.
"""
import numpy as np
import torch


def fibonacci_sphere(k: int, phase: float = 0.0, z_max: float = 0.75) -> np.ndarray:
    """(K,3) near-uniform directions, z in [-z_max, z_max].

    z_max=0.75 (±48 deg) suits the EVAL rig: pole views of a single-image
    reconstruction are unsupervised and look bad in both renders. TRAIN rigs
    must use a wider z_max (~0.95): any surface band no training view covers
    keeps its un-refined colours and shows up as visible patches at the
    refined/raw boundary (seen on the top of heads).
    phase: azimuth offset in radians — keeps a train rig disjoint from eval."""
    i = np.arange(k, dtype=np.float64)
    z = -z_max + 2.0 * z_max * (i + 0.5) / k
    phi = i * (np.pi * (3.0 - np.sqrt(5.0))) + phase  # golden angle
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1).astype(np.float32)


def look_at_w2c(eye: np.ndarray, target: np.ndarray, up_world=(0, 0, 1)):
    """World-to-camera (R, t) with camera +Z forward, +X right, +Y down."""
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    upw = np.asarray(up_world, np.float64)
    right = np.cross(fwd, upw)
    n = np.linalg.norm(right)
    if n < 1e-6:                                       # looking straight along up
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        n = np.linalg.norm(right)
    right = right / n
    down = np.cross(fwd, right)                        # +Y down (right-handed)
    R = np.stack([right, down, fwd], axis=0)           # rows = camera axes
    t = -R @ eye
    return R.astype(np.float32), t.astype(np.float32)


def make_rig(points: np.ndarray, k: int = 8, res: int = 512,
             fov_deg: float = 40.0, dist_factor: float = 1.5,
             phase: float = 0.0, z_max: float = 0.75) -> list:
    """Build K cameras around `points` (object geometry, Z-up frame).

    Returns list of dicts: {R (3,3), t (3,), fx, fy, cx, cy, res}.
    Distance = dist_factor x bbox diag, so the object fills ~60% of the frame.
    phase: azimuth offset — keeps a train rig disjoint from the eval rig.
    z_max: elevation coverage; use ~0.95 for TRAIN rigs (full surface coverage),
    default 0.75 for the held-out eval rig.
    """
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) * 0.5
    diag = float(np.linalg.norm(hi - lo))
    dist = dist_factor * max(diag, 1e-6) * 0.5 / np.tan(np.radians(fov_deg) * 0.5)

    f = 0.5 * res / np.tan(np.radians(fov_deg) * 0.5)
    cams = []
    for d in fibonacci_sphere(k, phase=phase, z_max=z_max):
        eye = center + d * dist
        R, t = look_at_w2c(eye.astype(np.float64), center.astype(np.float64))
        cams.append({"R": R, "t": t, "fx": f, "fy": f,
                     "cx": res * 0.5, "cy": res * 0.5, "res": res})
    return cams


def cam_to_torch(cam: dict, device) -> dict:
    out = dict(cam)
    out["R"] = torch.as_tensor(cam["R"], dtype=torch.float32, device=device)
    out["t"] = torch.as_tensor(cam["t"], dtype=torch.float32, device=device)
    return out

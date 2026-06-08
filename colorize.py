"""
Color projection for TripoSplat's Projection node.

Transfers per-vertex color from a colored 3D Gaussian-splat point cloud onto a
plain (grey) input mesh. The splat is produced by the same TripoSplat model from
the source image; the grey mesh comes from another generator (TripoSG, Trellis,
…), so the two differ in center / scale / orientation. We:

  1. rotate the splat from its native Z-up frame into glTF Y-up (the convention
     of the input mesh) — same matrix used by generator.py's reconstruction,
  2. coarse-align by centroid + bounding-box scale,
  3. optionally refine with PyMeshLab ICP,
  4. project color with a scipy cKDTree nearest-neighbour lookup (the same
     remap strategy generator.py already uses after decimation/hole-closing).

The 3DGS color is stored as SH band-0 (f_dc_0/1/2), NOT red/green/blue; the
caller passes already-decoded RGB, and the .ply reader below does the same
conversion (f_dc -> RGB with the C0 term, matching splat_mesh.py).
"""
import numpy as np

_C0 = 0.28209479177387814  # SH band-0 DC -> base RGB (matches splat_mesh.py)

# Splat native frame (Z-up) -> Modly/glTF (Y-up). Same proper rotation (det +1)
# generator.py applies to reconstructed vertices before export.
_SPLAT_TO_GLTF = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Alignment helpers
# --------------------------------------------------------------------------- #

def _bbox_diag(pts):
    return float(np.linalg.norm(pts.max(0) - pts.min(0)))


def _coarse_align(src_xyz, tgt_xyz):
    """Map source points into the target's frame: match centroid and bbox scale.
    PyMeshLab ICP does not normalize scale/center and diverges if the clouds are
    not already roughly superimposed, so this runs first either way."""
    src_c = src_xyz.mean(0)
    tgt_c = tgt_xyz.mean(0)
    src_d = _bbox_diag(src_xyz)
    tgt_d = _bbox_diag(tgt_xyz)
    scale = (tgt_d / src_d) if src_d > 1e-9 else 1.0
    return (src_xyz - src_c) * scale + tgt_c


def _apply_boost(col, mode):
    """Optional post-process on the transferred 0..1 colors. The viewer shows
    vertex color as-is (no linear->sRGB lift), so splat DC colors look dark.
      Vivid  - luminance-scaled saturation boost (same formula as splat_mesh.py),
               darks left neutral so eyes/shadows don't pick up a cast.
      Bright - gamma lift (<1) that raises midtones/whites so e.g. a white muzzle
               reads white instead of grey.
    """
    if mode in ("Vivid", "Vivid + Bright"):
        luma = (col @ np.array([0.299, 0.587, 0.114], np.float64))[:, None]
        amt = 0.5 * np.clip(luma / 0.4, 0.0, 1.0)
        col = np.clip(luma + (col - luma) * (1.0 + amt), 0.0, 1.0)
    if mode in ("Bright", "Vivid + Bright"):
        col = np.clip(col ** 0.7, 0.0, 1.0)
    return col


def _estimate_normals(pts, k=12):
    """Per-point normals via PCA over each point's k nearest neighbours (smallest
    eigenvector of the local covariance), oriented outward from the cloud centroid.
    Used to reject wrong-side matches in the color transfer."""
    from scipy.spatial import cKDTree
    k = min(k, len(pts))
    _, idx = cKDTree(pts).query(pts, k=k)              # [N,k]
    nbr = pts[idx]                                     # [N,k,3]
    nbr = nbr - nbr.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nbr, nbr) / k      # [N,3,3]
    _w, v = np.linalg.eigh(cov)                        # ascending eigenvalues
    normals = v[:, :, 0]                               # smallest -> surface normal
    flip = np.einsum('ni,ni->n', normals, pts - pts.mean(0)) < 0
    normals[flip] = -normals[flip]
    return normals


def _rot_y(theta):
    """Rotation matrix about the vertical (Y) axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _axis_aligned_rotations():
    """The 24 proper rotations of a cube (axis permutations with sign flips,
    det == +1). Used as orientation candidates so the alignment works for any
    input-mesh convention, not just glTF Y-up."""
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                M[row, col] = s
            if abs(np.linalg.det(M) - 1.0) < 1e-6:
                mats.append(M)
    return mats


def _auto_orient(src_xyz, tgt_xyz, n_sample=20000, accept_margin=0.85):
    """Robust orientation alignment for arbitrary input meshes — not every GLB
    shares the splat's up-axis or facing. Coarse bbox align fixes center + scale
    but not rotation, so here we recover rotation by minimizing mean
    nearest-neighbour distance to the target:

      1. test the 24 axis-aligned rotations (up-axis swaps + 90° turns),
      2. fine-refine yaw about the vertical (±45° / 3°) on top of the best one.

    Even a few degrees of residual yaw visibly misplaces fine features (an eye
    slides onto the cheek, ears rotate), hence the fine pass. Optimization:
    rotating src by R and matching the fixed probe is distance-equivalent to
    rotating the probe by R^-1 against the fixed src tree, so we build the src
    KD-tree once and only transform the small probe set.

    Guard: the caller has already applied the trusted fixed _SPLAT_TO_GLTF
    rotation (splat and mesh usually share the glTF convention), so 'no extra
    rotation' (identity) is the baseline. We only override it when the searched
    orientation beats identity by `accept_margin`. Without this, a near-symmetric
    object (e.g. a cat) locks onto a wrong-but-almost-as-good pose and the colors
    rotate (eye onto the cheek, scattered blotches)."""
    from scipy.spatial import cKDTree
    if len(tgt_xyz) > n_sample:
        sel = np.random.default_rng(0).choice(len(tgt_xyz), n_sample, replace=False)
        probe = tgt_xyz[sel]
    else:
        probe = tgt_xyz
    center = tgt_xyz.mean(0)
    tree = cKDTree(src_xyz)

    def cost(R):
        # nearest src to (R^-1 . probe); for row vectors p @ (R^-1).T == p @ R
        rp = (probe - center) @ R + center
        d, _ = tree.query(rp, k=1)
        return float(d.mean())

    c_identity = cost(np.eye(3))
    R0 = min(_axis_aligned_rotations(), key=cost)
    best_a = min(range(-45, 46, 3),
                 key=lambda a: cost(_rot_y(np.deg2rad(a)) @ R0))
    R = _rot_y(np.deg2rad(best_a)) @ R0
    # Keep the trusted fixed rotation unless the search is a clear win.
    if cost(R) >= accept_margin * c_identity:
        return src_xyz
    return (src_xyz - center) @ R.T + center


def _icp_refine(src_xyz, tgt_verts, iters=40, sample=20000, tol=1e-6):
    """Point-to-point ICP (Kabsch/SVD) refining a rigid src->tgt fit from the
    already good orientation produced by _auto_orient. Catches the residual tilt
    and translation that the axis-aligned + yaw search can't express. Estimates
    each step's transform from a src subsample matched to nearest target
    vertices, applies it to all source points, and stops once it converges.
    Non-fatal: returns the input unchanged on failure."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(tgt_verts)
        cur = np.asarray(src_xyz, dtype=np.float64).copy()
        if len(cur) > sample:
            sub = np.random.default_rng(0).choice(len(cur), sample, replace=False)
        else:
            sub = np.arange(len(cur))
        diag = _bbox_diag(tgt_verts)
        for _ in range(iters):
            P = cur[sub]
            _, idx = tree.query(P, k=1)
            Q = tgt_verts[idx]
            pc, qc = P.mean(0), Q.mean(0)
            H = (P - pc).T @ (Q - qc)
            U, _s, Vt = np.linalg.svd(H)
            D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T
            t = qc - R @ pc
            cur = cur @ R.T + t
            # converged once the step barely moves points (relative to mesh size)
            if diag > 0 and (np.linalg.norm(t) + abs(np.arccos(
                    np.clip((np.trace(R) - 1) / 2, -1, 1))) * diag) < tol * diag:
                break
        return cur
    except Exception as exc:
        print(f"[colorize] ICP refine skipped: {exc}")
        return np.asarray(src_xyz, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def colorize_from_points(mesh, src_xyz, src_rgb, opacity=None, use_icp=False,
                         pre_rotate=True, k=12, min_opacity=0.1, color_boost="Natural"):
    """Transfer per-vertex color from a colored point cloud onto `mesh`
    (a trimesh.Trimesh). `src_xyz` is Nx3, `src_rgb` is Nx3 in 0..1.

    Pipeline: drop near-transparent floater gaussians (by `opacity`) -> optional
    Z-up->Y-up rotation -> coarse bbox align -> auto yaw -> optional ICP ->
    inverse-distance-weighted k-NN color lookup. Every vertex gets a color (no
    grey fallback), so the mesh comes out fully colored.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src_xyz, dtype=np.float64)
    rgb = np.asarray(src_rgb, dtype=np.float64)

    # Near-transparent gaussians are volumetric floaters with arbitrary (often
    # dark) colors; spread through the volume, they get picked by the NN lookup
    # and muddy/darken the result. Keep only the opaque (surface) gaussians —
    # but guard against over-filtering an unusually faint splat.
    if opacity is not None:
        op = np.asarray(opacity, dtype=np.float64).reshape(-1)
        keep = op > min_opacity
        if keep.sum() >= max(100, int(0.05 * len(op))):
            src, rgb = src[keep], rgb[keep]

    if pre_rotate:
        src = src @ _SPLAT_TO_GLTF

    tgt = np.asarray(mesh.vertices, dtype=np.float64)
    if len(tgt) == 0:
        raise ValueError("Target mesh has no vertices")
    if len(src) == 0:
        raise ValueError("Source point cloud is empty")

    src = _coarse_align(src, tgt)
    src = _auto_orient(src, tgt)  # recover rotation (any GLB convention) + fine yaw
    if use_icp:
        src = _icp_refine(src, tgt)

    k = max(1, min(int(k), len(src)))
    dist, idx = cKDTree(src).query(tgt, k=k)
    if k == 1:
        colors = rgb[idx]
    else:
        # Inverse-distance weight of the k nearest splat colors (smooths speckle,
        # recovers flat regions like white better than a single point).
        w = 1.0 / (dist + 1e-8)                                    # [V,k]

        # Normal-aware gating: keep only neighbours whose surface faces the same
        # way as the vertex (dot of normals > 0). On a thin shell this discards
        # the points sitting on the *opposite* side that otherwise bleed their
        # color across (e.g. pink bow onto the cheek) and cause speckle.
        try:
            vn = np.asarray(mesh.vertex_normals, dtype=np.float64)  # [V,3]
            sn = _estimate_normals(src)                             # [S,3]
            cons = np.clip(np.einsum('vi,vki->vk', vn, sn[idx]), 0.0, None)
            wn = w * cons
            bad = wn.sum(axis=1) <= 1e-12   # no same-side neighbour -> keep plain w
            wn[bad] = w[bad]
            w = wn
        except Exception as exc:
            print(f"[colorize] normal-aware weighting skipped: {exc}")

        w /= w.sum(axis=1, keepdims=True)
        colors = (rgb[idx] * w[..., None]).sum(axis=1)

    colors = _apply_boost(colors, color_boost)

    mesh.visual.vertex_colors = np.concatenate(
        [np.clip(colors * 255.0, 0, 255).astype(np.uint8),
         np.full((len(colors), 1), 255, dtype=np.uint8)], axis=1)
    return mesh


def colorize_with_splat(mesh, splat_ply_path, use_icp=False):
    """Convenience wrapper: read a colored 3DGS .ply from disk, then colorize.
    Used when a splat .ply is wired in rather than generated in memory."""
    xyz, rgb = _read_splat_ply(splat_ply_path)
    return colorize_from_points(mesh, xyz, rgb, use_icp=use_icp, pre_rotate=True)


# --------------------------------------------------------------------------- #
# .ply reading (for the wired-splat path)
# --------------------------------------------------------------------------- #

_PLY_TYPE_MAP = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}


def _read_splat_ply(path):
    """Parse a binary/ascii .ply point cloud. Returns (xyz Nx3 float32, rgb Nx3
    float32 in 0..1). Reads f_dc_0/1/2 (3DGS SH) if present, else red/green/blue.
    Raises ValueError on an unreadable / colorless file."""
    with open(path, "rb") as fh:
        raw = fh.read()

    end_token = b"end_header\n"
    end = raw.find(end_token)
    if end == -1:
        raise ValueError("PLY: no end_header found")
    header = raw[:end].decode("ascii", errors="replace").splitlines()
    body = raw[end + len(end_token):]

    fmt = None
    n_vertices = 0
    props = []          # list of (name, numpy_dtype_str)
    in_vertex = False
    for line in header:
        toks = line.split()
        if not toks:
            continue
        key = toks[0]
        if key == "format":
            fmt = toks[1]                       # binary_little_endian / ascii / ...
        elif key == "element":
            in_vertex = toks[1] == "vertex"
            if in_vertex:
                n_vertices = int(toks[2])
        elif key == "property" and in_vertex:
            if toks[1] == "list":
                raise ValueError("PLY: list property in vertex element unsupported")
            ply_type, name = toks[1], toks[2]
            np_type = _PLY_TYPE_MAP.get(ply_type)
            if np_type is None:
                raise ValueError(f"PLY: unknown property type {ply_type!r}")
            props.append((name, np_type))

    if not props or n_vertices == 0:
        raise ValueError("PLY: no vertex properties / vertices")

    names = [p[0] for p in props]
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"PLY: missing coordinate '{axis}'")

    if fmt == "ascii":
        flat = np.fromstring(body.decode("ascii", errors="replace"),
                             sep=" ", dtype=np.float64)
        flat = flat[: n_vertices * len(props)].reshape(n_vertices, len(props))
        col = {name: flat[:, i] for i, (name, _) in enumerate(props)}
    else:
        endian = "<" if "little" in (fmt or "") else ">"
        dt = np.dtype([(name, endian + t) for name, t in props])
        arr = np.frombuffer(body, dtype=dt, count=n_vertices)
        col = {name: arr[name].astype(np.float64) for name, _ in props}

    xyz = np.stack([col["x"], col["y"], col["z"]], axis=1).astype(np.float32)

    if all(f"f_dc_{i}" in col for i in range(3)):
        f_dc = np.stack([col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]], axis=1)
        rgb = np.clip(f_dc * _C0 + 0.5, 0.0, 1.0).astype(np.float32)
    elif all(c in col for c in ("red", "green", "blue")):
        rgb = np.stack([col["red"], col["green"], col["blue"]], axis=1)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError("PLY: no color (neither f_dc_* nor red/green/blue)")

    return xyz, rgb

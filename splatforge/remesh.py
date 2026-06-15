"""
Botsch-Kobbelt style adaptive local remeshing: edge split, collapse, flip.

Reference algorithm: Botsch & Kobbelt 2004, "A Remeshing Approach to
Multiresolution Modeling"; Palfinger 2022, "Continuous Remeshing for Inverse
Rendering" — reimplemented from the published algorithm. No code copied.

Topology operations are in numpy (dynamic arrays). Curvature guidance (which
edges to split/collapse based on local |∇ρ|) is computed in torch via the
GaussianField passed to remesh_step(). All geometry decisions are local
(adjacent faces only) for O(E) complexity.
"""
import numpy as np
import torch


# ------------------------------------------------------------------ #
# Edge data structures
# ------------------------------------------------------------------ #

def _build_edges(faces: np.ndarray):
    """Unique edges and edge→face adjacency.

    Returns:
      edges      (E,2) int64, each row sorted so i < j
      edge_faces list[list[int]] — face indices adjacent to each edge
    """
    F = len(faces)
    raw = np.concatenate([
        np.sort(faces[:, [0, 1]], axis=1),
        np.sort(faces[:, [1, 2]], axis=1),
        np.sort(faces[:, [2, 0]], axis=1),
    ], axis=0)                                    # (3F, 2)
    face_of = np.tile(np.arange(F, dtype=np.int64), 3)

    edges, inv = np.unique(raw, axis=0, return_inverse=True)
    E = len(edges)

    ef = [[] for _ in range(E)]
    for k, fi in enumerate(face_of):
        ef[inv[k]].append(int(fi))

    return edges, ef


def _edge_lengths(verts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    d = verts[edges[:, 1]] - verts[edges[:, 0]]
    return np.linalg.norm(d, axis=1).astype(np.float32)


# ------------------------------------------------------------------ #
# Split
# ------------------------------------------------------------------ #

def split_edges(verts: np.ndarray, faces: np.ndarray,
                edges_to_split: np.ndarray) -> tuple:
    """Insert midpoint on each selected edge and re-triangulate adjacent faces.

    edges_to_split: (S,2) int64. Handles 0/1/2/3 splits per face.
    Returns (new_verts, new_faces).
    """
    if len(edges_to_split) == 0:
        return verts, faces

    split_map = {}
    new_v = verts.tolist()
    for e in edges_to_split:
        i, j = int(min(e[0], e[1])), int(max(e[0], e[1]))
        if (i, j) not in split_map:
            split_map[(i, j)] = len(new_v)
            new_v.append(((verts[i] + verts[j]) * 0.5).tolist())

    new_f = []
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        mab = split_map.get((min(a, b), max(a, b)))
        mbc = split_map.get((min(b, c), max(b, c)))
        mca = split_map.get((min(c, a), max(c, a)))
        n = (mab is not None) + (mbc is not None) + (mca is not None)
        if n == 0:
            new_f.append([a, b, c])
        elif n == 1:
            if   mab: new_f += [[a, mab, c], [mab, b,   c  ]]
            elif mbc: new_f += [[a, b,   mbc], [a,  mbc, c  ]]
            else:     new_f += [[a, b,   mca], [mca, b,  c  ]]
        elif n == 2:
            if   mab is None: new_f += [[a, b, mbc], [a, mbc, mca], [mbc, c, mca]]
            elif mbc is None: new_f += [[a, mab, mca], [mab, b, c], [mab, c, mca]]
            else:             new_f += [[a, mab, c], [mab, mbc, c], [mab, b, mbc]]
        else:  # 1→4 subdivision
            new_f += [[a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca]]

    return (np.array(new_v, dtype=np.float32),
            np.array(new_f, dtype=np.int64))


# ------------------------------------------------------------------ #
# Collapse
# ------------------------------------------------------------------ #

def collapse_short_edges(verts: np.ndarray, faces: np.ndarray,
                         edges: np.ndarray, edge_faces: list,
                         elen: np.ndarray, threshold: float) -> tuple:
    """Collapse interior edges shorter than threshold to their midpoint.

    Boundary edges (< 2 adjacent faces) are skipped.
    No inversion check for performance; degenerate faces cleaned up after.
    Returns (new_verts, new_faces).
    """
    short = np.where(elen < threshold)[0]
    if len(short) == 0:
        return verts, faces

    short = short[np.argsort(elen[short])]   # shortest first
    verts = verts.copy()
    faces = faces.copy()
    removed: set = set()

    for ei in short:
        i, j = int(edges[ei, 0]), int(edges[ei, 1])
        if i in removed or j in removed:
            continue
        if len(edge_faces[ei]) != 2:          # boundary → skip
            continue
        verts[i] = (verts[i] + verts[j]) * 0.5
        faces[faces == j] = i
        removed.add(j)

    if not removed:
        return verts, faces

    # Remove degenerate faces (two or more vertices equal)
    valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[valid]

    # Compact vertex array
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used].astype(np.float32), remap[faces].astype(np.int64)


# ------------------------------------------------------------------ #
# Flip
# ------------------------------------------------------------------ #

def flip_edges(verts: np.ndarray, faces: np.ndarray,
               edges: np.ndarray, edge_faces: list) -> np.ndarray:
    """Flip interior edges to reduce vertex valence deviation from 6.
    Returns new faces array (verts unchanged).
    """
    faces = faces.copy()
    valence = np.zeros(len(verts), dtype=np.int32)
    for f in faces:
        valence[f[0]] += 1
        valence[f[1]] += 1
        valence[f[2]] += 1

    for ei in range(len(edges)):
        if len(edge_faces[ei]) != 2:
            continue
        fi0, fi1 = edge_faces[ei][0], edge_faces[ei][1]
        a, b = int(edges[ei, 0]), int(edges[ei, 1])
        f0, f1 = faces[fi0], faces[fi1]

        c_cands = [int(v) for v in f0 if v != a and v != b]
        d_cands = [int(v) for v in f1 if v != a and v != b]
        if not c_cands or not d_cands:
            continue
        c, d = c_cands[0], d_cands[0]

        # Valence improvement
        cur = abs(valence[a]-6) + abs(valence[b]-6) + abs(valence[c]-6) + abs(valence[d]-6)
        nw  = abs(valence[a]-7) + abs(valence[b]-7) + abs(valence[c]+1-6) + abs(valence[d]+1-6)
        if nw >= cur:
            continue

        # Geometry validity: new triangles (a,c,d) and (b,d,c) must not invert
        n0   = np.cross(verts[c] - verts[a], verts[d] - verts[a])
        n1   = np.cross(verts[d] - verts[b], verts[c] - verts[b])
        orig = np.cross(verts[int(f0[1])] - verts[int(f0[0])],
                        verts[int(f0[2])] - verts[int(f0[0])])
        if (np.linalg.norm(n0) < 1e-12 or np.linalg.norm(n1) < 1e-12
                or np.dot(n0, orig) < 0 or np.dot(n1, orig) < 0):
            continue

        faces[fi0] = [a, c, d]
        faces[fi1] = [b, d, c]
        valence[a] -= 1; valence[b] -= 1
        valence[c] += 1; valence[d] += 1

    return faces


# ------------------------------------------------------------------ #
# Full remesh pass
# ------------------------------------------------------------------ #

def remesh_step(verts: np.ndarray, faces: np.ndarray,
                field=None,
                h_factor: float = 1.0,
                alpha: float = 2.0,
                do_split: bool = True,
                do_collapse: bool = True,
                do_flip: bool = True) -> tuple:
    """One Botsch-Kobbelt remesh pass: split → collapse → flip.

    h_factor: global edge-length target scale (1.0 = current mean).
    alpha: curvature-adaptation strength (0 = uniform target).
             Higher alpha → shorter target in high-|∇ρ| regions.
    field: GaussianField used for curvature guidance; None = uniform.
    Returns (new_verts, new_faces) in the same coordinate frame.
    """
    edges, ef = _build_edges(faces)
    elen = _edge_lengths(verts, edges)
    h_t = float(elen.mean()) * h_factor

    # Curvature-adaptive per-edge target edge length
    if alpha > 0 and field is not None:
        v_t = torch.as_tensor(verts, device=field.device)
        with torch.no_grad():
            g_mag = field.density_grad(v_t).norm(dim=1).cpu().numpy()  # (V,)
        g_max = g_mag.max() + 1e-8
        h_v = h_t / (1.0 + alpha * g_mag / g_max)
        # Per-edge target = min of the two endpoint targets (conservative)
        h_edge = np.minimum(h_v[edges[:, 0]], h_v[edges[:, 1]])
    else:
        h_edge = np.full(len(edges), h_t, dtype=np.float32)

    if do_split:
        mask = elen > (4.0 / 3.0) * h_edge
        verts, faces = split_edges(verts, faces, edges[mask])
        edges, ef = _build_edges(faces)
        elen = _edge_lengths(verts, edges)

    if do_collapse:
        h_c = float(elen.mean()) * (4.0 / 5.0)
        verts, faces = collapse_short_edges(verts, faces, edges, ef, elen, h_c)
        edges, ef = _build_edges(faces)

    if do_flip:
        faces = flip_edges(verts, faces, edges, ef)

    return verts, faces

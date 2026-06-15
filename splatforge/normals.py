"""Vertex-normal smoothing — single source of truth for the GLB export
(generator._smooth_vertex_normals) AND the LIT regression harness
(debug/lit_regression.py), so the harness validates the exact operator that ships.

Why this exists: the area-weighted normals of a high-res Surface-Nets mesh carry
the grid's sub-voxel relief noise. UNLIT that is invisible (flat DC colour); the
Modly viewer LIGHTS the mesh from the stored normals, so the noise shades as
mottling/orange-peel. Smoothing the normals (positions untouched) fixes it.

A UNIFORM Laplacian also softens genuine creases — that is the see-saw: tune it to
clean the smooth duck and it smears the cat's muzzle/chin. The BILATERAL filter
weights each neighbour by normal similarity, so high-frequency noise is averaged
out while real creases (low cos across a sharp edge -> low weight) are preserved.
Same operator, locally adaptive: no per-object tuning, no see-saw.

All functions are numpy on (V,3)/(F,3) arrays.
"""
import numpy as np
import scipy.sparse as sp


def _directed_edges(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)            # both directions
    return e[:, 0].astype(np.int64), e[:, 1].astype(np.int64)


def area_weighted_normals(verts, faces):
    """Smooth area-weighted vertex normals (same field trimesh/the viewer start from)."""
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    for j in range(3):
        np.add.at(vn, faces[:, j], fn)
    n = vn / np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)
    return n.astype(np.float32)


def smooth_normals(verts, faces, normals0=None, iters=20, mode="bilateral",
                   sigma=0.30):
    """Feature-preserving vertex-normal smoothing (positions untouched).

    mode='laplacian': uniform neighbour averaging — kills noise AND softens creases.
    mode='bilateral': neighbour weight exp(-(1-cos)/sigma); averages noise but keeps
      genuine creases. Smaller sigma = more crease-preserving (sharper).
    """
    verts = np.asarray(verts, np.float32)
    faces = np.asarray(faces, np.int64)
    V = len(verts)
    n = area_weighted_normals(verts, faces) if normals0 is None \
        else np.asarray(normals0, np.float32).copy()
    if V == 0 or len(faces) == 0:
        return n
    src, dst = _directed_edges(faces)

    if mode == "laplacian":
        A = sp.coo_matrix((np.ones(len(src), np.float32), (src, dst)),
                          shape=(V, V)).tocsr()
        deg = np.asarray(A.sum(1)).ravel().clip(min=1.0)[:, None]
        for _ in range(iters):
            n = 0.5 * (n + (A @ n) / deg)
            n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)
        return n.astype(np.float32)

    for _ in range(iters):                            # bilateral
        cos = np.clip((n[src] * n[dst]).sum(1), -1.0, 1.0)
        w = np.exp(-(1.0 - cos) / sigma).astype(np.float32)
        A = sp.coo_matrix((w, (src, dst)), shape=(V, V)).tocsr()
        den = np.asarray(A.sum(1)).ravel() + 1.0      # +1.0 = self weight
        n = ((A @ n) + n) / den[:, None]              # weighted neighbours + self
        n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)
    return n.astype(np.float32)

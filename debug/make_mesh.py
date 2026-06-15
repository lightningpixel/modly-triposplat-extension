"""Generate an 81k-face icosphere, sanitize it (Test 1), save raw + sanitized npz."""
import sys, numpy as np, trimesh
out = sys.argv[1]
m = trimesh.creation.icosphere(subdivisions=6, radius=1.0)
v = np.ascontiguousarray(m.vertices, np.float32)
f = np.ascontiguousarray(m.faces, np.uint32)
print(f"raw: v={len(v)} f={len(f)} finite={np.isfinite(v).all()} "
      f"dup_v={len(v)-len(np.unique(v,axis=0))}")

# Test-1 sanitation via pymeshlab.
import pymeshlab as ml
ms = ml.MeshSet()
ms.add_mesh(ml.Mesh(vertex_matrix=v.astype(np.float64), face_matrix=f.astype(np.int32)))
for filt in ("meshing_remove_duplicate_vertices", "meshing_remove_duplicate_faces",
             "meshing_remove_null_faces", "meshing_remove_unreferenced_vertices"):
    try:
        getattr(ms, filt)()
    except Exception as e:
        print(f"  {filt}: {e}")
cm = ms.current_mesh()
vs = np.ascontiguousarray(cm.vertex_matrix(), np.float32)
fs = np.ascontiguousarray(cm.face_matrix(), np.uint32)
print(f"sanitized: v={len(vs)} f={len(fs)}")
np.savez(out, v_raw=v, f_raw=f, v_san=vs, f_san=fs)
print("saved", out)

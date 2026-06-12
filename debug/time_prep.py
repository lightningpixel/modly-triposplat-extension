"""Time the solidify+decimate prep at several resample resolutions on the real mesh."""
import sys, time, glob, numpy as np, trimesh, pymeshlab as ml
glb = [g for g in sorted(glob.glob(r"C:\Users\LORCHIE\Documents\Modly\extensions\triposplat\_full_test_out\*.glb")) if "tex" not in g][0]
m = trimesh.load(glb, force="mesh")
V = np.ascontiguousarray(m.vertices, np.float64); F = np.ascontiguousarray(m.faces, np.int32)
print(f"src faces {len(F)}", flush=True)

def open_edges(faces):
    e = np.sort(faces[:, [0,1,1,2,2,0]].reshape(-1,2), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True); return int((c==1).sum())

TARGET = 50000
for res in (224, 192, 160, 128):
    t = time.time()
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(vertex_matrix=V, face_matrix=F))
    ms.generate_resampled_uniform_mesh(cellsize=ml.PercentageValue(100.0/res),
                                       offset=ml.PercentageValue(50.0), mergeclosevert=True)
    nf_solid = ms.current_mesh().face_number()
    if nf_solid > TARGET:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=TARGET, preservetopology=True)
    cm = ms.current_mesh()
    ff = np.ascontiguousarray(cm.face_matrix(), np.int64)
    vv = np.ascontiguousarray(cm.vertex_matrix(), np.float32)
    g = trimesh.Trimesh(vertices=vv, faces=ff, process=True); g.merge_vertices()
    print(f"res {res:>3}: prep {time.time()-t:5.1f}s  solid_faces {nf_solid:>7} -> {len(ff):>6}  "
          f"real_holes {open_edges(g.faces)}", flush=True)

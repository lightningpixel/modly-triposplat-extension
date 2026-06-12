"""Irregular high-poly source (sub=7 decimated) at targets >80k to find the real ceiling."""
import sys, numpy as np, trimesh, pymeshlab as ml
m = trimesh.creation.icosphere(subdivisions=7, radius=1.0)  # 327680 faces
v0 = np.ascontiguousarray(m.vertices, np.float64); f0 = np.ascontiguousarray(m.faces, np.int32)
for tgt in (80000, 120000, 160000, 220000):
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(vertex_matrix=v0, face_matrix=f0))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=tgt, preservetopology=True)
    cm = ms.current_mesh()
    vv = np.ascontiguousarray(cm.vertex_matrix(), np.float32)
    ff = np.ascontiguousarray(cm.face_matrix(), np.uint32)
    np.savez(sys.argv[1].replace("TGT", str(tgt)), v_san=vv, f_san=ff)
    print(f"tgt={tgt} -> v={len(vv)} f={len(ff)}", flush=True)

"""Decimate the 81k icosphere to several target face counts; save one npz each."""
import sys, numpy as np, pymeshlab as ml
src = np.load(sys.argv[1])
v = src["v_san"].astype(np.float64); f = src["f_san"].astype(np.int32)
for tgt in (30000, 45000, 60000, 70000):
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=v, face_matrix=f))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=tgt, preservetopology=True)
    cm = ms.current_mesh()
    vv = np.ascontiguousarray(cm.vertex_matrix(), np.float32)
    ff = np.ascontiguousarray(cm.face_matrix(), np.uint32)
    out = sys.argv[2].replace("TGT", str(tgt))
    np.savez(out, v_san=vv, f_san=ff)
    print(f"tgt={tgt} -> v={len(vv)} f={len(ff)}  {out}", flush=True)

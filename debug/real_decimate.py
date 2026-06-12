"""Decimate a REAL TripoSplat mesh (non-manifold dual-contour) to several targets."""
import sys, glob, numpy as np, trimesh, pymeshlab as ml
glb = sorted(glob.glob(r"C:\Users\LORCHIE\Documents\Modly\extensions\triposplat\_full_test_out\*.glb"))[0]
m = trimesh.load(glb, force="mesh")
v = np.ascontiguousarray(m.vertices, np.float64); f = np.ascontiguousarray(m.faces, np.int32)
print(f"source {glb.split(chr(92))[-1]}: v={len(v)} f={len(f)}", flush=True)
for tgt in (60000, 40000, 30000, 20000, 12000):
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(vertex_matrix=v, face_matrix=f))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=tgt)
    cm = ms.current_mesh()
    vv = np.ascontiguousarray(cm.vertex_matrix(), np.float32)
    ff = np.ascontiguousarray(cm.face_matrix(), np.uint32)
    np.savez(sys.argv[1].replace("TGT", str(tgt)), verts=vv, faces=ff)
    print(f"tgt={tgt} -> v={len(vv)} f={len(ff)}", flush=True)

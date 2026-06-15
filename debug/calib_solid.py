"""Solidify a real mesh via pymeshlab uniform resampling (watertight manifold, no
skimage), decimate to targets, save npz."""
import sys, glob, numpy as np, trimesh, pymeshlab as ml
glb = [g for g in sorted(glob.glob(r"C:\Users\LORCHIE\Documents\Modly\extensions\triposplat\_full_test_out\*.glb")) if "tex" not in g][0]
m = trimesh.load(glb, force="mesh")
print(f"source: v={len(m.vertices)} f={len(m.faces)} watertight={m.is_watertight}", flush=True)

ms = ml.MeshSet()
ms.add_mesh(ml.Mesh(vertex_matrix=np.ascontiguousarray(m.vertices, np.float64),
                    face_matrix=np.ascontiguousarray(m.faces, np.int32)))
try:
    ms.generate_resampled_uniform_mesh(cellsize=ml.PercentageValue(0.45),
                                       offset=ml.PercentageValue(50.0), mergeclosevert=True)
except Exception as e:
    print("PARAM ERROR:", e); sys.exit(2)
cm = ms.current_mesh()
sv = np.ascontiguousarray(cm.vertex_matrix(), np.float64)
sf = np.ascontiguousarray(cm.face_matrix(), np.int32)
solid = trimesh.Trimesh(vertices=sv, faces=sf, process=False)
print(f"solidified: v={len(sv)} f={len(sf)} watertight={solid.is_watertight}", flush=True)

for tgt in (50000, 60000, 70000):
    ms2 = ml.MeshSet(); ms2.add_mesh(ml.Mesh(vertex_matrix=sv, face_matrix=sf))
    ms2.meshing_decimation_quadric_edge_collapse(targetfacenum=tgt, preservetopology=True)
    dm = ms2.current_mesh()
    vv = np.ascontiguousarray(dm.vertex_matrix(), np.float32)
    ff = np.ascontiguousarray(dm.face_matrix(), np.uint32)
    tm = trimesh.Trimesh(vertices=vv, faces=ff, process=False)
    np.savez(sys.argv[1].replace("TGT", str(tgt)), verts=vv, faces=ff)
    print(f"tgt={tgt} -> v={len(vv)} f={len(ff)} watertight={tm.is_watertight}", flush=True)

"""Time xatlas on a {verts,faces} npz (subprocess-safe). Exit 0=ok."""
import sys, time, numpy as np, xatlas
d = np.load(sys.argv[1])
v = np.ascontiguousarray(d["verts"], np.float32)
f = np.ascontiguousarray(d["faces"], np.uint32)
t = time.time()
vmap, idx, uv = xatlas.parametrize(v, f)
print(f"OK f={len(f)} {time.time()-t:.1f}s verts_out={len(vmap)}", flush=True)

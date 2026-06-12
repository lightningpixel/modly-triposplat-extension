"""Subprocess unwrap worker: load npz, run xatlas on the chosen array set.
Exit 0 = OK, segfault = non-zero/crash exit code (the discriminating signal)."""
import sys, time, numpy as np, xatlas
npz, which = sys.argv[1], sys.argv[2]   # which: "raw" or "san"
d = np.load(npz)
v = np.ascontiguousarray(d[f"v_{which}"], np.float32)
f = np.ascontiguousarray(d[f"f_{which}"], np.uint32)
print(f"unwrap {which}: v={len(v)} f={len(f)} finite={np.isfinite(v).all()}", flush=True)
t = time.time()
vmap, idx, uv = xatlas.parametrize(v, f)
print(f"OK {time.time()-t:.1f}s verts_out={len(vmap)} uv[{uv.min():.3f},{uv.max():.3f}]", flush=True)

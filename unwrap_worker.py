"""Subprocess UV-unwrap worker.

The xatlas wheel segfaults (Windows access violation) on meshes above a size/topology
threshold — a native crash that would take down the whole Modly backend if called
in-process. Running it here, in a throwaway subprocess, turns that crash into a plain
non-zero exit the caller can catch and fall back from.

Reads {verts (N,3) f32, faces (F,3) u32} from an .npz, writes
{vmapping, indices, uvs} to an .npz. xatlas duplicates vertices along UV seams, so
vmapping maps each output vertex back to an input vertex.

Chart/pack options are tuned for the fuzzy organic surfaces this pipeline produces:
the parametrize() defaults shatter a bumpy volumetric remesh into thousands of
micro-islands packed with ZERO padding — at viewer mip levels every island bleeds
into its neighbours (dirty-speckle look). Fewer/larger charts + an 8 px gutter keep
the texture clean under minification. Override via argv[3] (JSON), e.g.
'{"max_cost": 4.0, "padding": 4}'.
"""
import json
import sys
import numpy as np
import xatlas


def main():
    d = np.load(sys.argv[1])
    v = np.ascontiguousarray(d["verts"], np.float32)
    f = np.ascontiguousarray(d["faces"], np.uint32)
    opt = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}

    co = xatlas.ChartOptions()
    # merge aggressively: bumpy normals on the remeshed surface are noise, not
    # genuine chart boundaries
    co.max_cost = float(opt.get("max_cost", 8.0))
    co.normal_deviation_weight = float(opt.get("normal_deviation_weight", 0.5))
    co.max_iterations = int(opt.get("max_iterations", 2))
    po = xatlas.PackOptions()
    po.padding = int(opt.get("padding", 8))      # gutter: survives ~3 mip levels
    po.bilinear = True

    atlas = xatlas.Atlas()
    atlas.add_mesh(v, f)
    atlas.generate(chart_options=co, pack_options=po)
    vmapping, indices, uvs = atlas[0]
    uvs = np.asarray(uvs, np.float32)
    if uvs.size and float(uvs.max()) > 1.5:      # pixel-space output -> normalize
        uvs = uvs / [max(atlas.width, 1), max(atlas.height, 1)]
    np.savez(sys.argv[2],
             vmapping=np.asarray(vmapping, np.int64),
             indices=np.asarray(indices, np.int64),
             uvs=uvs.astype(np.float32))


if __name__ == "__main__":
    main()

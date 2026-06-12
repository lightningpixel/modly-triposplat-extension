"""Subprocess UV-unwrap worker.

The xatlas wheel segfaults (Windows access violation) on meshes above a size/topology
threshold — a native crash that would take down the whole Modly backend if called
in-process. Running it here, in a throwaway subprocess, turns that crash into a plain
non-zero exit the caller can catch and fall back from.

Reads {verts (N,3) f32, faces (F,3) u32} from an .npz, writes
{vmapping, indices, uvs} to an .npz. xatlas duplicates vertices along UV seams, so
vmapping maps each output vertex back to an input vertex.
"""
import sys
import numpy as np
import xatlas


def main():
    d = np.load(sys.argv[1])
    v = np.ascontiguousarray(d["verts"], np.float32)
    f = np.ascontiguousarray(d["faces"], np.uint32)
    vmapping, indices, uvs = xatlas.parametrize(v, f)
    np.savez(sys.argv[2],
             vmapping=np.asarray(vmapping, np.int64),
             indices=np.asarray(indices, np.int64),
             uvs=np.asarray(uvs, np.float32))


if __name__ == "__main__":
    main()

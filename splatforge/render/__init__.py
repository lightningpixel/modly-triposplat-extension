"""Phase B rendering stack: cameras, splat reference renderer, mesh rasterizer, metrics."""
from .cameras import make_rig, cam_to_torch
from .splat import render_splats
from .mesh import render_mesh, sample_texture
from .metrics import view_metrics, composite

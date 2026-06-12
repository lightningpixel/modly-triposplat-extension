"""
Competition selector — delegates to whichever agent's implementation won bench.py.
After Phase A gate approval: replace this file with the winner's full code.
"""

# Phase A gate verdict (2026-06-11, 3 objects, self-contained npz): FAILED.
# All variants (ALPHA / BETA / HYBRID) worsen chamfer vs Standard even with the
# field aligned (0.9×spacing floor + surface tau cut degradation 49% → 6-9%, but
# the gap is structural — see splatforge/pipeline.py docstring). Phase A is closed;
# the geometry preset was removed from the manifest. Kept for Phase B reuse.
_WINNER = "alpha"

DEFAULT_WEIGHTS = {
    "Refined": None,   # filled in by the winner's module
    "Maximum": None,
}


def optimize_phase_a(verts, faces, field, n_steps, lr, weights, device, budget_s):
    if _WINNER == "alpha":
        from .competition.optimize_alpha import optimize_phase_a as _fn
    else:
        from .competition.optimize_beta import optimize_phase_a as _fn
    return _fn(verts, faces, field, n_steps, lr, weights, device, budget_s)


def get_default_weights(preset: str) -> dict:
    if _WINNER == "alpha":
        from .competition.optimize_alpha import DEFAULT_WEIGHTS as _dw
    else:
        from .competition.optimize_beta import DEFAULT_WEIGHTS as _dw
    return _dw.get(preset, list(_dw.values())[0])

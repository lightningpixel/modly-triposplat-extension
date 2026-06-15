"""
PSNR / SSIM in pure torch (no skimage/lpips dependency).

Both metrics are computed over the UNION of the two foreground masks, composited
on a fixed grey background: comparing over the whole frame would reward agreeing
on empty background; intersection-only would hide silhouette errors — the union
penalizes coverage mismatch exactly where one render has content and the other
doesn't, which is the failure mode the Phase B gate must see.
"""
import torch
import torch.nn.functional as F

_BG = 0.5  # composite background — identical for both renders


def composite(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """(H,W,3),(H,W) -> (H,W,3) over the fixed grey background."""
    return rgb * alpha[..., None] + _BG * (1.0 - alpha[..., None])


def psnr_masked(img_a, img_b, mask) -> float:
    """PSNR over masked pixels. img (H,W,3) in [0,1], mask (H,W) bool."""
    if not bool(mask.any()):
        return float("nan")
    d2 = ((img_a - img_b) ** 2)[mask]
    mse = float(d2.mean())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def _gauss_kernel(ks: int = 11, sigma: float = 1.5, device="cpu"):
    x = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = (g / g.sum())
    return (g[:, None] * g[None, :]).reshape(1, 1, ks, ks)


def ssim_masked(img_a, img_b, mask, ks: int = 11) -> float:
    """Mean SSIM over masked pixels. Standard Wang et al. constants."""
    if not bool(mask.any()):
        return float("nan")
    dev = img_a.device
    a = img_a.permute(2, 0, 1)[None]                    # (1,3,H,W)
    b = img_b.permute(2, 0, 1)[None]
    k = _gauss_kernel(ks, device=dev).expand(3, 1, ks, ks)
    pad = ks // 2

    mu_a = F.conv2d(a, k, padding=pad, groups=3)
    mu_b = F.conv2d(b, k, padding=pad, groups=3)
    s_aa = F.conv2d(a * a, k, padding=pad, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, k, padding=pad, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, k, padding=pad, groups=3) - mu_a * mu_b

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / \
           ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
    ssim = ssim.mean(1)[0]                              # (H,W) channel-mean
    return float(ssim[mask].mean())


def view_metrics(splat_rgb, splat_alpha, mesh_rgb, mesh_alpha,
                 fg_thresh: float = 0.5,
                 splat_depth=None, mesh_depth=None) -> dict:
    """All Phase B per-view metrics between a splat render and a mesh render.

    If both depth maps are given, also returns depth_rmse over the INTERSECTION
    of the masks (scene units): the splat depth is smooth across the surface, so
    high-frequency mesh geometry noise (crevasses) shows up directly here while
    being nearly invisible in the unlit colour metrics.
    """
    sm = splat_alpha > fg_thresh
    mm = mesh_alpha > fg_thresh
    mask = sm | mm
    a = composite(splat_rgb, splat_alpha)
    b = composite(mesh_rgb, mesh_alpha)
    out = {
        "psnr": psnr_masked(a, b, mask),
        "ssim": ssim_masked(a, b, mask),
        "iou":  float((sm & mm).sum()) / max(float(mask.sum()), 1.0),
    }
    if splat_depth is not None and mesh_depth is not None:
        inter = sm & mm
        if bool(inter.any()):
            d = (splat_depth - mesh_depth)[inter]
            out["depth_rmse"] = float(torch.sqrt((d ** 2).mean()))
        else:
            out["depth_rmse"] = float("nan")
    return out

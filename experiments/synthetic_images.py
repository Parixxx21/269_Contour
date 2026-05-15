"""
Synthetic test image generation.

Each test case is a dict:
  image      : (H,W) float ndarray in [0,1]
  mask       : (H,W) bool ndarray (ground-truth segmentation)
  gt_contour : (M,2) ndarray of (row,col) boundary points
  name       : human-readable label
"""

import numpy as np
from skimage.measure import find_contours
from utils.image_utils import add_gaussian_noise, add_salt_pepper, reduce_contrast


# ------------------------------------------------------------------
# Primitive shape generators
# ------------------------------------------------------------------

def _disk(shape, center, radius):
    y, x = np.mgrid[: shape[0], : shape[1]]
    return ((y - center[0]) ** 2 + (x - center[1]) ** 2) < radius ** 2


def _ellipse(shape, center, a, b, angle=0.0):
    y, x = np.mgrid[: shape[0], : shape[1]]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr = cos_a * (x - center[1]) + sin_a * (y - center[0])
    yr = -sin_a * (x - center[1]) + cos_a * (y - center[0])
    return (xr / a) ** 2 + (yr / b) ** 2 < 1


def _star(shape, center, r_outer, r_inner, n_points=5):
    y, x = np.mgrid[: shape[0], : shape[1]]
    theta = np.arctan2(y - center[0], x - center[1])
    dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    r_star = r_inner + (r_outer - r_inner) * (np.cos(n_points * theta) + 1) / 2
    return dist < r_star


def _mask_to_image(mask):
    """Binary mask → float image with slight interior gradient."""
    return mask.astype(float)


def _get_contour(mask):
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    return max(contours, key=len)   # longest contour


# ------------------------------------------------------------------
# Test case factory
# ------------------------------------------------------------------

def generate_test_cases(shape=(256, 256), seed=42):
    """
    Returns an ordered dict of test cases, each with increasing difficulty.

    Cases
    -----
    1. clean_disk            — baseline, easy
    2. noisy_disk            — Gaussian noise σ=0.15
    3. low_contrast_ellipse  — contrast factor 0.2
    4. noisy_lc_ellipse      — low contrast + Gaussian noise σ=0.10
    5. noisy_star            — non-convex shape + noise
    """
    rng_seed = seed
    cy, cx = shape[0] // 2, shape[1] // 2
    cases = {}

    # 1. Clean disk
    mask = _disk(shape, (cy, cx), radius=60)
    img = _mask_to_image(mask)
    cases["clean_disk"] = dict(
        image=img, mask=mask, gt_contour=_get_contour(mask), name="Clean Disk"
    )

    # 2. Noisy disk
    img_noisy = add_gaussian_noise(img, sigma=0.15, seed=rng_seed)
    cases["noisy_disk"] = dict(
        image=img_noisy, mask=mask, gt_contour=_get_contour(mask), name="Noisy Disk (σ=0.15)"
    )

    # 3. Low-contrast ellipse
    mask_e = _ellipse(shape, (cy, cx), a=75, b=45, angle=np.pi / 6)
    img_e = reduce_contrast(_mask_to_image(mask_e), factor=0.2)
    cases["low_contrast_ellipse"] = dict(
        image=img_e, mask=mask_e, gt_contour=_get_contour(mask_e),
        name="Low-Contrast Ellipse (factor=0.2)"
    )

    # 4. Noisy low-contrast ellipse (hardest standard case)
    img_e_noisy = add_gaussian_noise(img_e, sigma=0.10, seed=rng_seed + 1)
    cases["noisy_lc_ellipse"] = dict(
        image=img_e_noisy, mask=mask_e, gt_contour=_get_contour(mask_e),
        name="Noisy Low-Contrast Ellipse"
    )

    # 5. Noisy star (non-convex)
    mask_s = _star(shape, (cy, cx), r_outer=65, r_inner=35, n_points=5)
    img_s = add_gaussian_noise(_mask_to_image(mask_s), sigma=0.12, seed=rng_seed + 2)
    cases["noisy_star"] = dict(
        image=img_s, mask=mask_s, gt_contour=_get_contour(mask_s), name="Noisy Star (non-convex)"
    )

    return cases

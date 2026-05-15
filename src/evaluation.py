import numpy as np
from scipy.spatial.distance import cdist
from skimage.draw import polygon


def snake_to_mask(snake, shape):
    """Rasterize a closed contour into a binary mask."""
    mask = np.zeros(shape, dtype=bool)
    rr, cc = polygon(snake[:, 0], snake[:, 1], shape)
    mask[rr, cc] = True
    return mask


def compute_iou(snake, gt_mask, shape):
    """Intersection-over-Union between predicted and ground-truth masks."""
    pred = snake_to_mask(snake, shape)
    inter = np.logical_and(pred, gt_mask).sum()
    union = np.logical_or(pred, gt_mask).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_hausdorff(snake, gt_contour):
    """Symmetric Hausdorff distance between two contour point sets."""
    D = cdist(snake, gt_contour)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def compute_mean_dist(snake, gt_contour):
    """Mean of nearest-neighbor distances from predicted to GT contour."""
    D = cdist(snake, gt_contour)
    return float(D.min(axis=1).mean())


def evaluate_snake(snake, gt_mask, gt_contour, shape):
    """
    Compute all evaluation metrics for a fitted snake.

    Returns
    -------
    dict with keys: iou, hausdorff, mean_dist
    """
    return {
        "iou": compute_iou(snake, gt_mask, shape),
        "hausdorff": compute_hausdorff(snake, gt_contour),
        "mean_dist": compute_mean_dist(snake, gt_contour),
    }

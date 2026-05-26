"""
Run the classical snake on local real-image datasets under data/.

Supported layouts:
  - Ultrasound-style pairs: data/train/<id>.tif and data/train/<id>_mask.tif
  - CTC-style HeLa pairs: data/Fluo-N2DL-HeLa/<seq>/tNNN.tif and
    data/Fluo-N2DL-HeLa/<seq>_GT/SEG/man_segNNN.tif

The mask is used only for evaluation and for a controlled initial contour.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import io
from skimage.measure import find_contours
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_snake
from src.snake import ClassicalSnake
from utils.visualization import plot_comparison, plot_convergence, plot_metrics_bar


def load_image(path, target_shape=None, is_mask=False):
    """Load a grayscale image or mask, optionally resizing to target_shape."""
    try:
        img = io.imread(path)
    except ValueError as exc:
        if "imagecodecs" not in str(exc):
            raise
        with Image.open(path) as pil_img:
            img = np.array(pil_img)
    if img.ndim == 3:
        img = img[..., 0]

    if is_mask:
        mask = img > 0
        if target_shape is not None and mask.shape != target_shape:
            mask = resize(
                mask.astype(float), target_shape, order=0,
                anti_aliasing=False, preserve_range=True,
            ) > 0.5
        return mask

    img = img.astype(float)
    img -= img.min()
    img /= img.max() + 1e-8
    if target_shape is not None and img.shape != target_shape:
        img = resize(img, target_shape, anti_aliasing=True, preserve_range=True)
    return img


def mask_to_contour(mask):
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    return max(contours, key=len)


def init_from_mask(mask, margin=20, n_points=100):
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return init_center(mask.shape, n_points=n_points)

    cy = 0.5 * (rows.min() + rows.max())
    cx = 0.5 * (cols.min() + cols.max())
    radius = 0.5 * max(rows.max() - rows.min(), cols.max() - cols.min()) + margin
    radius = np.clip(radius, 12, 0.48 * min(mask.shape))

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    snake = np.column_stack([cy + radius * np.sin(theta), cx + radius * np.cos(theta)])
    snake[:, 0] = np.clip(snake[:, 0], 0, mask.shape[0] - 1)
    snake[:, 1] = np.clip(snake[:, 1], 0, mask.shape[1] - 1)
    return snake


def init_center(shape, radius=None, n_points=100):
    cy, cx = shape[0] / 2, shape[1] / 2
    radius = radius or 0.35 * min(shape)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([cy + radius * np.sin(theta), cx + radius * np.cos(theta)])


def ultrasound_pairs(root):
    train_dir = Path(root) / "train"
    if not train_dir.exists():
        return []

    pairs = []
    for image_path in sorted(train_dir.glob("*.tif")):
        if image_path.stem.endswith("_mask"):
            continue
        mask_path = image_path.with_name(f"{image_path.stem}_mask.tif")
        if mask_path.exists():
            pairs.append((f"ultrasound_{image_path.stem}", image_path, mask_path))
    return pairs


def hela_pairs(root):
    root = Path(root) / "Fluo-N2DL-HeLa"
    if not root.exists():
        return []

    pairs = []
    pattern = re.compile(r"man_seg(\d+)\.tif$")
    for seg_dir in sorted(root.glob("*_GT/SEG")):
        seq = seg_dir.parent.name.removesuffix("_GT")
        image_dir = root / seq
        if not image_dir.exists():
            continue
        for mask_path in sorted(seg_dir.glob("man_seg*.tif")):
            match = pattern.match(mask_path.name)
            if not match:
                continue
            frame = match.group(1)
            image_path = image_dir / f"t{frame}.tif"
            if image_path.exists():
                pairs.append((f"hela_{seq}_{frame}", image_path, mask_path))
    return pairs


def discover_pairs(root, dataset):
    if dataset == "ultrasound":
        return ultrasound_pairs(root)
    if dataset == "hela":
        return hela_pairs(root)
    return ultrasound_pairs(root) + hela_pairs(root)


def run(root="data", output_dir="results/classical_data", dataset="auto",
        limit=10, target_shape=(256, 256), init_mode="mask"):
    os.makedirs(output_dir, exist_ok=True)
    pairs = discover_pairs(root, dataset)
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError(f"No supported image/mask pairs found under {root}")

    results = {}
    for case_id, image_path, mask_path in pairs:
        image = load_image(image_path, target_shape=target_shape)
        mask = load_image(mask_path, target_shape=image.shape, is_mask=True)
        gt_contour = mask_to_contour(mask)
        if gt_contour is None:
            print(f"Skipping {case_id}: empty mask")
            continue

        init = init_from_mask(mask) if init_mode == "mask" else init_center(image.shape)
        model = ClassicalSnake()

        start = time.time()
        snake, history = model.fit(image, init)
        elapsed = time.time() - start
        metrics = evaluate_snake(snake, mask, gt_contour, image.shape)
        results[case_id] = {"Classical": metrics}

        print(
            f"{case_id:<24} IoU={metrics['iou']:.3f} "
            f"Haus={metrics['hausdorff']:.1f} MCD={metrics['mean_dist']:.2f} "
            f"({elapsed:.1f}s)"
        )

        prefix = os.path.join(output_dir, case_id)
        snakes = {"Classical": snake}
        histories = {"Classical": history}

        fig = plot_comparison(
            image, snakes, gt_contour=gt_contour,
            title=case_id, save_path=f"{prefix}_comparison.png"
        )
        plt.close(fig)

        fig = plot_metrics_bar({"Classical": metrics}, save_path=f"{prefix}_metrics.png")
        plt.close(fig)

        fig = plot_convergence(image, histories, save_path=f"{prefix}_convergence.png")
        plt.close(fig)

    write_summary(results, output_dir)


def write_summary(results, output_dir):
    lines = ["case,iou,hausdorff,mean_dist"]
    for case_id, result in results.items():
        metrics = result["Classical"]
        lines.append(
            f"{case_id},{metrics['iou']:.6f},"
            f"{metrics['hausdorff']:.6f},{metrics['mean_dist']:.6f}"
        )
    path = os.path.join(output_dir, "summary.csv")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved classical data results to {output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ClassicalSnake on local data images")
    parser.add_argument("--root", default="data")
    parser.add_argument("--output-dir", default="results/classical_data")
    parser.add_argument("--dataset", choices=["auto", "ultrasound", "hela"], default="auto")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--target-shape", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--init-mode", choices=["mask", "center"], default="mask")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        root=args.root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        limit=args.limit,
        target_shape=tuple(args.target_shape),
        init_mode=args.init_mode,
    )

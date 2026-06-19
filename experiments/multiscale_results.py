"""
MultiscaleSnake evaluation:
  Coarse-to-fine multi-scale snake with fixed parameters

Sections
--------
  1. Synthetic test cases (5 canonical shapes)
  2. Real data — Fluo-N2DL-HeLa

Usage
-----
    python experiments/multiscale_results.py
    python experiments/multiscale_results.py --n-real 20
    python experiments/multiscale_results.py --skip-synth
    python experiments/multiscale_results.py --skip-real
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tifffile
from skimage.measure import find_contours

from experiments.synthetic_images import generate_test_cases
from src.multiscale_snake import MultiscaleSnake
from src.evaluation import evaluate_snake


OUT_DIR = "results/multiscale_results"
COLOR = "#FFD400"

SYNTH_N_ITERS = 5000
REAL_N_ITERS = 8000

CASE_ORDER = [
    "clean_disk",
    "noisy_disk",
    "low_contrast_ellipse",
    "noisy_lc_ellipse",
    "noisy_star",
]
CASE_LABELS = {
    "clean_disk": "Clean Disk",
    "noisy_disk": "Noisy Disk",
    "low_contrast_ellipse": "Low-Contrast Ellipse",
    "noisy_lc_ellipse": "Noisy LC Ellipse",
    "noisy_star": "Noisy Star",
}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def contour_init(mask, n=120, scale=1.3):
    """Circular snake centred at mask centroid, radius = scale × RMS distance to centroid."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return circular_init(mask.shape, n=n)
    cy, cx = ys.mean(), xs.mean()
    r = scale * np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).mean())
    r = max(r, 20.0)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = np.clip(cy + r * np.sin(theta), 2, mask.shape[0] - 2)
    cols = np.clip(cx + r * np.cos(theta), 2, mask.shape[1] - 2)
    return np.column_stack([rows, cols])


def circular_init(shape, n=120, scale=0.35):
    """Default circular initializer for synthetic benchmarks."""
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    r = scale * min(h, w)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = np.clip(cy + r * np.sin(theta), 2, h - 2)
    cols = np.clip(cx + r * np.cos(theta), 2, w - 2)
    return np.column_stack([rows, cols])


def load_fluo_hela(n_images=20):
    """Load up to n_images cell samples from Fluo-N2DL-HeLa (sequences 01 + 02)."""
    base = os.path.join(_ROOT, "data", "Fluo-N2DL-HeLa", "Fluo-N2DL-HeLa")
    samples = []
    for seq in ["01", "02"]:
        seg_dir = os.path.join(base, f"{seq}_GT", "SEG")
        img_dir = os.path.join(base, seq)
        if not os.path.isdir(seg_dir):
            continue
        for seg_file in sorted(os.listdir(seg_dir)):
            if not seg_file.endswith(".tif"):
                continue
            frame = seg_file.replace("man_seg", "").replace(".tif", "")
            img_path = os.path.join(img_dir, f"t{frame}.tif")
            if not os.path.exists(img_path):
                continue
            img_raw = tifffile.imread(img_path).astype(np.float32)
            img_norm = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-8)
            seg = tifffile.imread(os.path.join(seg_dir, seg_file))
            for label in np.unique(seg):
                if label == 0:
                    continue
                cell_mask = seg == label
                contours = find_contours(cell_mask.astype(float), 0.5)
                if not contours:
                    continue
                samples.append({
                    "image": img_norm,
                    "mask": cell_mask,
                    "gt_contour": contours[0],
                    "name": f"HeLa_seq{seq}_frame{frame}_cell{label}",
                })
                if len(samples) >= n_images:
                    print(f"  Loaded {len(samples)} Fluo-N2DL-HeLa samples")
                    return samples
    print(f"  Loaded {len(samples)} Fluo-N2DL-HeLa samples")
    return samples


def load_ultrasound(n_images=20):
    """Load up to n_images nerve samples from Ultrasound Nerve Segmentation dataset."""
    base = os.path.join(_ROOT, "data", "ultrasound-nerve-segmentation", "train")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Ultrasound data not found: {base}")
    files = sorted(f for f in os.listdir(base) if f.endswith(".tif") and "_mask" not in f)
    samples = []
    for fname in files:
        mask_path = os.path.join(base, fname.replace(".tif", "_mask.tif"))
        if not os.path.exists(mask_path):
            continue
        mask_raw = tifffile.imread(mask_path)
        if mask_raw.max() == 0:
            continue
        img_raw = tifffile.imread(os.path.join(base, fname)).astype(np.float32)
        img_norm = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-8)
        cell_mask = mask_raw > 0
        contours = find_contours(cell_mask.astype(float), 0.5)
        if not contours:
            continue
        samples.append({
            "image": img_norm,
            "mask": cell_mask,
            "gt_contour": max(contours, key=len),
            "name": fname.replace(".tif", ""),
        })
        if len(samples) >= n_images:
            break
    if not samples:
        raise FileNotFoundError(f"No valid ultrasound samples with non-empty masks in {base}")
    print(f"  Loaded {len(samples)} ultrasound samples")
    return samples


def build_multiscale(n_iter):
    return MultiscaleSnake(
        alpha=0.015,
        beta=0.1,
        gamma=5.0,
        sigma=8.0,
        n_levels=3,
        n_iter=n_iter,
        reparam_every=50,
        update_every=20,
        wline=0.0,
        wedge=1.0,
        boundary_condition="periodic",
    )


def run_synthetic():
    cases = generate_test_cases()
    cases = {k: cases[k] for k in CASE_ORDER if k in cases}

    model = build_multiscale(SYNTH_N_ITERS)
    results = {}
    snakes = {}

    for key, case in cases.items():
        img = case["image"]
        mask = case["mask"]
        gt_c = case["gt_contour"]
        init = circular_init(img.shape)

        snake, _ = model.fit(img, init.copy())
        results[key] = evaluate_snake(snake, mask, gt_c, img.shape)
        snakes[key] = snake
        print(
            f"  [synth] {key}  IoU={results[key]['iou']:.3f}  "
            f"Hausdorff={results[key]['hausdorff']:.1f}  "
            f"MeanDist={results[key]['mean_dist']:.2f}"
        )

    return cases, results, snakes


def run_real(samples, tag, n_iters=None):
    if not samples:
        print(f"  No samples for {tag}.")
        return [], []

    model = build_multiscale(n_iters or REAL_N_ITERS)
    results = []
    vis_samples = []

    for i, sample in enumerate(samples):
        img = sample["image"]
        mask = sample["mask"]
        gt_c = sample["gt_contour"]
        name = sample["name"]
        init = contour_init(mask)

        snake, _ = model.fit(img, init.copy())
        metrics = evaluate_snake(snake, mask, gt_c, img.shape)
        results.append(metrics)

        if len(vis_samples) < 5:
            vis_samples.append({
                "image": img,
                "mask": mask,
                "gt_contour": gt_c,
                "name": name,
                "snake": snake,
            })

        print(
            f"  [{tag} {i+1}/{len(samples)}] {name}  "
            f"IoU={metrics['iou']:.3f}  "
            f"Hausdorff={metrics['hausdorff']:.1f}  "
            f"MeanDist={metrics['mean_dist']:.2f}"
        )

    return vis_samples, results


def plot_synthetic(cases, results, snakes, out_dir):
    fig, axes = plt.subplots(1, len(CASE_ORDER), figsize=(3.5 * len(CASE_ORDER), 3.5))
    for ci, key in enumerate(CASE_ORDER):
        ax = axes[ci]
        img = cases[key]["image"]
        gt_c = cases[key]["gt_contour"]
        iou = results[key]["iou"]
        ax.imshow(img, cmap="gray")
        ax.plot(gt_c[:, 1], gt_c[:, 0], "w--", lw=1.2, alpha=0.6)
        ax.plot(snakes[key][:, 1], snakes[key][:, 0], color=COLOR, lw=1.8)
        ax.set_title(f"{CASE_LABELS[key]}\nIoU={iou:.3f}", fontsize=9, color=COLOR)
        ax.axis("off")

    fig.suptitle(
        f"Synthetic Benchmark — MultiscaleSnake ({SYNTH_N_ITERS} iters)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "synthetic_visual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    lines = [
        f"Synthetic benchmark  ({SYNTH_N_ITERS} iters)\n",
        f"{'Case':<26}  {'IoU':>6}  {'Hausdorff':>10}  {'MeanDist':>10}",
        "-" * 58,
    ]
    for key in CASE_ORDER:
        row = results[key]
        lines.append(
            f"{CASE_LABELS[key]:<26}  {row['iou']:>6.3f}  "
            f"{row['hausdorff']:>10.2f}  {row['mean_dist']:>10.2f}"
        )

    path = os.path.join(out_dir, "synthetic_table.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Table: {path}")


def plot_real(vis_samples, results, tag, out_dir, n_iters):
    if not results:
        return

    if vis_samples:
        n_vis = len(vis_samples)
        fig, axes = plt.subplots(1, n_vis, figsize=(3.5 * n_vis, 3.5))
        if n_vis == 1:
            axes = [axes]
        for ci, sample in enumerate(vis_samples):
            ax = axes[ci]
            gt_c = sample["gt_contour"]
            iou = results[ci]["iou"]
            ax.imshow(sample["image"], cmap="gray")
            ax.plot(gt_c[:, 1], gt_c[:, 0], "w--", lw=1.2, alpha=0.6)
            ax.plot(sample["snake"][:, 1], sample["snake"][:, 0], color=COLOR, lw=1.8)
            ax.set_title(
                f"{sample['name'].split('_cell')[0][-12:]}\nIoU={iou:.3f}",
                fontsize=8,
                color=COLOR,
            )
            ax.axis("off")
        fig.suptitle(f"Real Data ({tag}) — MultiscaleSnake", fontsize=11, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"{tag}_visual.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    ious = [row["iou"] for row in results]
    hds = [row["hausdorff"] for row in results]
    mds = [row["mean_dist"] for row in results]
    lines = [
        f"Real data: {tag}  (n={len(results)}, {n_iters} iters)\n",
        f"{'Method':<14}  {'IoU mean±std':<20}  {'Hausdorff':<20}  MeanDist",
        "-" * 72,
        f"{'Multiscale':<14}  {np.mean(ious):.3f}±{np.std(ious):.3f}            "
        f"{np.mean(hds):.1f}±{np.std(hds):.1f}            "
        f"{np.mean(mds):.2f}±{np.std(mds):.2f}",
    ]
    path = os.path.join(out_dir, f"{tag}_table.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Table: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-real", type=int, default=20)
    parser.add_argument("--skip-synth", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_synth:
        print("\n=== Synthetic benchmark ===")
        cases, syn_results, syn_snakes = run_synthetic()
        plot_synthetic(cases, syn_results, syn_snakes, args.out_dir)

    if not args.skip_real:
        print(f"\n=== Real data: Fluo-HeLa (n={args.n_real}, {REAL_N_ITERS} iters) ===")
        fluo_samples = load_fluo_hela(n_images=args.n_real)
        vis_samples, fluo_results = run_real(fluo_samples, "fluo", n_iters=REAL_N_ITERS)
        plot_real(vis_samples, fluo_results, "fluo", args.out_dir, REAL_N_ITERS)

        try:
            print(f"\n=== Real data: Ultrasound (n={args.n_real}, {REAL_N_ITERS} iters) ===")
            us_samples = load_ultrasound(n_images=args.n_real)
            vis_samples, us_results = run_real(us_samples, "ultrasound", n_iters=REAL_N_ITERS)
            plot_real(vis_samples, us_results, "ultrasound", args.out_dir, REAL_N_ITERS)
        except FileNotFoundError as e:
            print(f"  Skipping ultrasound: {e}")


if __name__ == "__main__":
    main()

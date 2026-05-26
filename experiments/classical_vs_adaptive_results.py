"""
Generate Classical vs Adaptive snake result figures.

Outputs:
    results/classical_vs_adaptive_results/
        synthetic_visual.png
        synthetic_metrics.png
        synthetic_table.txt
        fluo_visual.png
        fluo_metrics.png
        fluo_table.txt
        ultrasound_visual.png
        ultrasound_metrics.png
        ultrasound_table.txt
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import io
from skimage.measure import find_contours

from experiments.synthetic_images import generate_test_cases
from src.adaptive_snake import AdaptiveSnake
from src.evaluation import evaluate_snake
from src.snake import ClassicalSnake


OUT_DIR = "results/classical_vs_adaptive_results"
METHODS = ["Classical", "Adaptive"]
COLORS = {
    "Classical": "#2196F3",
    "Adaptive": "#4CAF50",
}

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

SYNTH_N_ITERS = 1600
REAL_N_ITERS = 2500
ULTRASOUND_N_ITERS = 3500

ROOT = Path(__file__).resolve().parents[1]


def read_tif(path):
    try:
        return io.imread(path)
    except ValueError as exc:
        if "imagecodecs" not in str(exc):
            raise
        with Image.open(path) as img:
            return np.array(img)


def normalize_image(image):
    image = image.astype(float)
    image -= image.min()
    image /= image.max() + 1e-8
    return image


def contour_init(mask, n=120, scale=1.3):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        h, w = mask.shape
        cy, cx = h / 2, w / 2
        radius = 0.35 * min(mask.shape)
    else:
        cy, cx = ys.mean(), xs.mean()
        radius = scale * np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).mean())
        radius = max(radius, 20.0)

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = np.clip(cy + radius * np.sin(theta), 2, mask.shape[0] - 2)
    cols = np.clip(cx + radius * np.cos(theta), 2, mask.shape[1] - 2)
    return np.column_stack([rows, cols])


def circular_init(shape, radius=80, n=120):
    cy, cx = shape[0] // 2, shape[1] // 2
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cy + radius * np.sin(theta), cx + radius * np.cos(theta)])


def build_models(n_iter):
    return {
        "Classical": ClassicalSnake(n_iter=n_iter, update_every=20, reparam_every=0),
        "Adaptive": AdaptiveSnake(
            alpha=0.015,
            beta_min=0.005,
            beta_max=0.3,
            k=5.0,
            gamma=5.0,
            sigma=8.0,
            n_iter=n_iter,
            update_every=20,
            reparam_every=0,
            wedge=1.0,
        ),
    }


def load_fluo_hela(n_images):
    samples = []
    base = ROOT / "data" / "Fluo-N2DL-HeLa"

    for seq in ["01", "02"]:
        image_dir = base / seq
        seg_dir = base / f"{seq}_GT" / "SEG"
        if not image_dir.exists() or not seg_dir.exists():
            continue

        for seg_path in sorted(seg_dir.glob("man_seg*.tif")):
            frame = seg_path.stem.replace("man_seg", "")
            image_path = image_dir / f"t{frame}.tif"
            if not image_path.exists():
                continue

            image = normalize_image(read_tif(image_path))
            seg = read_tif(seg_path)
            for label in np.unique(seg):
                if label == 0:
                    continue
                mask = seg == label
                contours = find_contours(mask.astype(float), 0.5)
                if not contours:
                    continue
                samples.append({
                    "image": image,
                    "mask": mask,
                    "gt_contour": max(contours, key=len),
                    "name": f"HeLa_seq{seq}_frame{frame}_cell{int(label)}",
                })
                if len(samples) >= n_images:
                    return samples

    return samples


def load_ultrasound(n_images):
    samples = []
    train_dir = ROOT / "data" / "train"
    if not train_dir.exists():
        return samples

    image_paths = sorted(
        p for p in train_dir.glob("*.tif")
        if not p.stem.endswith("_mask")
    )
    for image_path in image_paths:
        mask_path = image_path.with_name(f"{image_path.stem}_mask.tif")
        if not mask_path.exists():
            continue
        mask_raw = read_tif(mask_path)
        if mask_raw.max() == 0:
            continue

        image = normalize_image(read_tif(image_path))
        mask = mask_raw > 0
        contours = find_contours(mask.astype(float), 0.5)
        if not contours:
            continue

        samples.append({
            "image": image,
            "mask": mask,
            "gt_contour": max(contours, key=len),
            "name": image_path.stem,
        })
        if len(samples) >= n_images:
            break

    return samples


def run_synthetic():
    cases = generate_test_cases()
    cases = {k: cases[k] for k in CASE_ORDER if k in cases}
    models = build_models(SYNTH_N_ITERS)
    results = {m: {} for m in METHODS}
    snakes = {m: {} for m in METHODS}

    for key, case in cases.items():
        init = circular_init(case["image"].shape)
        for method, model in models.items():
            snake, _ = model.fit(case["image"], init.copy())
            results[method][key] = evaluate_snake(
                snake, case["mask"], case["gt_contour"], case["image"].shape
            )
            snakes[method][key] = snake
        print(
            f"  [synthetic] {key}  "
            + "  ".join(f"{m}={results[m][key]['iou']:.3f}" for m in METHODS)
        )

    return cases, results, snakes


def run_real(samples, tag, n_iter):
    models = build_models(n_iter)
    all_results = {m: [] for m in METHODS}
    vis_samples = []

    for idx, sample in enumerate(samples):
        init = contour_init(sample["mask"])
        row = {
            "name": sample["name"],
            "image": sample["image"],
            "gt_contour": sample["gt_contour"],
            "snakes": {},
        }
        for method, model in models.items():
            snake, _ = model.fit(sample["image"], init.copy())
            metrics = evaluate_snake(
                snake, sample["mask"], sample["gt_contour"], sample["image"].shape
            )
            all_results[method].append(metrics)
            row["snakes"][method] = snake

        if len(vis_samples) < 5:
            vis_samples.append(row)
        print(
            f"  [{tag} {idx + 1}/{len(samples)}] {sample['name']}  "
            + "  ".join(f"{m}={all_results[m][-1]['iou']:.3f}" for m in METHODS)
        )

    return vis_samples, all_results


def plot_synthetic(cases, results, snakes, out_dir):
    fig, axes = plt.subplots(len(METHODS), len(CASE_ORDER),
                             figsize=(3.5 * len(CASE_ORDER), 3.2 * len(METHODS)))

    for col, key in enumerate(CASE_ORDER):
        image = cases[key]["image"]
        gt = cases[key]["gt_contour"]
        axes[0, col].set_title(CASE_LABELS[key], fontsize=9, fontweight="bold")
        for row, method in enumerate(METHODS):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray")
            ax.plot(gt[:, 1], gt[:, 0], "w--", lw=1.2, alpha=0.6)
            snake = snakes[method][key]
            ax.plot(snake[:, 1], snake[:, 0], color=COLORS[method], lw=1.8)
            ax.set_title(f"IoU={results[method][key]['iou']:.3f}",
                         fontsize=8, color=COLORS[method])
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(method, fontsize=9, fontweight="bold", color=COLORS[method])

    fig.suptitle("Synthetic Benchmark - Classical vs Adaptive", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "synthetic_visual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_metric_groups(
        CASE_ORDER, {k: CASE_LABELS[k] for k in CASE_ORDER},
        {m: [results[m][k] for k in CASE_ORDER] for m in METHODS},
        "Synthetic Metrics", os.path.join(out_dir, "synthetic_metrics.png")
    )
    write_table(
        "Synthetic benchmark", CASE_ORDER, {k: CASE_LABELS[k] for k in CASE_ORDER},
        {m: [results[m][k] for k in CASE_ORDER] for m in METHODS},
        os.path.join(out_dir, "synthetic_table.txt")
    )


def plot_real(vis_samples, all_results, tag, out_dir):
    if not all_results["Classical"]:
        return

    if vis_samples:
        fig, axes = plt.subplots(len(METHODS), len(vis_samples),
                                 figsize=(3.5 * len(vis_samples), 3.2 * len(METHODS)))
        if len(vis_samples) == 1:
            axes = axes[:, np.newaxis]

        for col, sample in enumerate(vis_samples):
            axes[0, col].set_title(sample["name"][-18:], fontsize=8, fontweight="bold")
            for row, method in enumerate(METHODS):
                ax = axes[row, col]
                ax.imshow(sample["image"], cmap="gray")
                gt = sample["gt_contour"]
                snake = sample["snakes"][method]
                ax.plot(gt[:, 1], gt[:, 0], "w--", lw=1.2, alpha=0.6)
                ax.plot(snake[:, 1], snake[:, 0], color=COLORS[method], lw=1.8)
                ax.set_title(f"IoU={all_results[method][col]['iou']:.3f}",
                             fontsize=8, color=COLORS[method])
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(method, fontsize=9, fontweight="bold", color=COLORS[method])

        fig.suptitle(f"{tag.title()} - Classical vs Adaptive", fontsize=12, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"{tag}_visual.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    plot_real_metric_summary(
        all_results, f"{tag.title()} Metrics", os.path.join(out_dir, f"{tag}_metrics.png")
    )
    write_real_table(all_results, tag, os.path.join(out_dir, f"{tag}_table.txt"))


def plot_metric_groups(case_keys, labels, result_lists, title, save_path):
    metrics = [("iou", "IoU up"), ("hausdorff", "Hausdorff down"), ("mean_dist", "Mean Dist down")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(case_keys))
    width = 0.35

    for ax, (metric, metric_title) in zip(axes, metrics):
        for idx, method in enumerate(METHODS):
            vals = [result_lists[method][i][metric] for i in range(len(case_keys))]
            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, vals, width, color=COLORS[method], label=method, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_title(metric_title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[k] for k in case_keys], rotation=20, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_real_metric_summary(all_results, title, save_path):
    metrics = [("iou", "IoU up"), ("hausdorff", "Hausdorff down"), ("mean_dist", "Mean Dist down")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    x = np.arange(len(METHODS))
    for ax, (metric, metric_title) in zip(axes, metrics):
        means = [np.mean([r[metric] for r in all_results[m]]) for m in METHODS]
        stds = [np.std([r[metric] for r in all_results[m]]) for m in METHODS]
        bars = ax.bar(x, means, yerr=stds, color=[COLORS[m] for m in METHODS],
                      capsize=5, alpha=0.85)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(metric_title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=15, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_table(title, case_keys, labels, result_lists, save_path):
    lines = [title, "", f"{'Case':<24} {'Classical IoU':>14} {'Adaptive IoU':>14}"]
    lines.append("-" * 56)
    for i, key in enumerate(case_keys):
        lines.append(
            f"{labels[key]:<24} "
            f"{result_lists['Classical'][i]['iou']:>14.3f} "
            f"{result_lists['Adaptive'][i]['iou']:>14.3f}"
        )
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def write_real_table(all_results, tag, save_path):
    lines = [f"Real data: {tag}", "", f"{'Method':<12} {'IoU mean±std':<18} {'Hausdorff':<18} MeanDist"]
    lines.append("-" * 68)
    for method in METHODS:
        ious = [r["iou"] for r in all_results[method]]
        hds = [r["hausdorff"] for r in all_results[method]]
        mds = [r["mean_dist"] for r in all_results[method]]
        lines.append(
            f"{method:<12} "
            f"{np.mean(ious):.3f}±{np.std(ious):.3f}        "
            f"{np.mean(hds):.1f}±{np.std(hds):.1f}        "
            f"{np.mean(mds):.2f}±{np.std(mds):.2f}"
        )
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-real", type=int, default=5)
    parser.add_argument("--skip-synth", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_synth:
        print("\n=== Synthetic ===")
        cases, results, snakes = run_synthetic()
        plot_synthetic(cases, results, snakes, args.out_dir)

    if not args.skip_real:
        print(f"\n=== Fluo-HeLa real data (n={args.n_real}) ===")
        fluo_samples = load_fluo_hela(args.n_real)
        vis, results = run_real(fluo_samples, "fluo", REAL_N_ITERS)
        plot_real(vis, results, "fluo", args.out_dir)

        print(f"\n=== Ultrasound real data (n={args.n_real}) ===")
        ultrasound_samples = load_ultrasound(args.n_real)
        vis, results = run_real(ultrasound_samples, "ultrasound", ULTRASOUND_N_ITERS)
        plot_real(vis, results, "ultrasound", args.out_dir)

    print(f"\nSaved Classical vs Adaptive results to {args.out_dir}/")


if __name__ == "__main__":
    main()

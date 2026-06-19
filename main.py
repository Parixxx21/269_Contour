"""
Quick-start demo: run a single test case and display results interactively.

Usage
-----
    python main.py                        # noisy disk (default)
    python main.py --case noisy_lc_ellipse
    python main.py --case noisy_star --show
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from src.snake import ClassicalSnake
from src.adaptive_snake import AdaptiveSnake
from src.multiscale_snake import MultiscaleSnake
from src.combined_snake import CombinedSnake
from src.evaluation import evaluate_snake
from experiments.synthetic_images import generate_test_cases
from utils.visualization import plot_comparison, plot_all_methods_overlay, plot_metrics_bar

VALID_CASES = [
    "clean_disk",
    "noisy_disk",
    "low_contrast_ellipse",
    "noisy_lc_ellipse",
    "noisy_star",
]


def circular_init(shape, radius=80, n_points=100):
    cy, cx = shape[0] // 2, shape[1] // 2
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([cy + radius * np.sin(theta),
                            cx + radius * np.cos(theta)])


def main():
    parser = argparse.ArgumentParser(description="Active contour demo")
    parser.add_argument("--case", default="noisy_disk", choices=VALID_CASES)
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively (requires display)")
    args = parser.parse_args()

    cases = generate_test_cases()
    case = cases[args.case]
    image = case["image"]
    shape = image.shape
    gt_mask = case["mask"]
    gt_contour = case["gt_contour"]

    print(f"Running demo on: {case['name']}")
    init = circular_init(shape)

    models = {
        "Classical": ClassicalSnake(alpha=0.015, beta=0.1, gamma=0.001,
                                    sigma=2.0, n_iter=2500),
        "Adaptive":  AdaptiveSnake(alpha=0.015, beta_min=0.005, beta_max=0.3,
                                   k=5.0, gamma=5.0, sigma=8.0, n_iter=2000,
                                   reparam_every=50),
        "Multiscale": MultiscaleSnake(alpha=0.015, beta=0.1, gamma=5.0,
                                      sigma=8.0, n_levels=3, n_iter=2500,
                                      reparam_every=50),
        "Combined":  CombinedSnake(alpha=0.015, beta_min=0.005, beta_max=0.3,
                                   k=5.0, gamma=5.0, sigma=8.0, n_iter=2000,
                                   reparam_every=50, n_levels=3),
    }

    results, snakes = {}, {}
    for name, model in models.items():
        snake, _ = model.fit(image, init.copy())
        metrics = evaluate_snake(snake, gt_mask, gt_contour, shape)
        results[name] = metrics
        snakes[name] = snake
        print(f"  {name:<12}  IoU={metrics['iou']:.3f}  "
              f"Hausdorff={metrics['hausdorff']:.1f}  "
              f"MeanDist={metrics['mean_dist']:.2f}")

    fig1 = plot_comparison(image, snakes, gt_contour=gt_contour,
                           title=case["name"])
    fig2 = plot_all_methods_overlay(image, snakes, gt_contour=gt_contour,
                                    title=case["name"])
    fig3 = plot_metrics_bar(results)

    fig1.savefig(f"results/demo_{args.case}_comparison.png",
                 dpi=150, bbox_inches="tight")
    fig2.savefig(f"results/demo_{args.case}_overlay.png",
                 dpi=150, bbox_inches="tight")
    fig3.savefig(f"results/demo_{args.case}_metrics.png",
                 dpi=150, bbox_inches="tight")
    print(f"\nSaved figures to results/")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()

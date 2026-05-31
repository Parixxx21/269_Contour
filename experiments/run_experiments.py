"""
Main experiment runner.

Compares four methods on five synthetic test cases and saves:
  results/<case>_comparison.png   — side-by-side snake outputs
  results/<case>_overlay.png      — all methods on one image
  results/<case>_metrics.png      — bar chart of IoU / Hausdorff / MCD
  results/summary.txt             — full numeric results table
"""

import os
import sys
import time
import numpy as np

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snake import ClassicalSnake
from src.adaptive_snake import AdaptiveSnake
from src.multiscale_snake import MultiscaleSnake
from src.combined_snake import CombinedSnake
from src.evaluation import evaluate_snake
from experiments.synthetic_images import generate_test_cases
from utils.visualization import (
    plot_comparison,
    plot_metrics_bar,
    plot_all_methods_overlay,
    plot_convergence,
)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Shared initialization helper
# ------------------------------------------------------------------

def circular_init(shape, center=None, radius=80, n_points=100):
    """Circular snake initialization, slightly larger than the target object."""
    if center is None:
        center = (shape[0] // 2, shape[1] // 2)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    rows = center[0] + radius * np.sin(theta)
    cols = center[1] + radius * np.cos(theta)
    return np.column_stack([rows, cols])


# ------------------------------------------------------------------
# Method factory
# ------------------------------------------------------------------

def build_methods():
    """Return dict of method_name → snake model instance."""
    return {
        "Classical": ClassicalSnake(
            alpha=0.015, beta=0.1, gamma=0.001, sigma=2.0, n_iter=2500,
            wedge=1.0
        ),
        "Adaptive": AdaptiveSnake(
            alpha=0.015, beta_min=0.005, beta_max=0.3, k=5.0,
            gamma=5.0, sigma=8.0, n_iter=2500, update_every=20, wedge=1.0
        ),
        "Multiscale": MultiscaleSnake(
            alpha=0.015, beta=0.1, gamma=0.001, sigma_coarse=4.0,
            sigma_fine=1.5, n_levels=3, n_iter=2500, wedge=1.0
        ),
        "Combined": CombinedSnake(
            alpha=0.015, beta_min=0.005, beta_max=0.3, k=5.0,
            gamma=5.0, sigma=8.0, n_iter=2500, update_every=20,
            reparam_every=50, n_levels=3, wedge=1.0
        ),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    test_cases = generate_test_cases(shape=(256, 256))
    all_results = {}

    for case_name, case in test_cases.items():
        image = case["image"]
        gt_mask = case["mask"]
        gt_contour = case["gt_contour"]
        shape = image.shape

        print(f"\n{'='*55}")
        print(f"  Case: {case['name']}")
        print(f"{'='*55}")

        init_snake = circular_init(shape, radius=80)
        methods = build_methods()

        case_results = {}
        case_snakes = {}
        case_histories = {}

        for name, model in methods.items():
            t0 = time.time()
            snake, history = model.fit(image, init_snake.copy())
            elapsed = time.time() - t0
            metrics = evaluate_snake(snake, gt_mask, gt_contour, shape)
            case_results[name] = metrics
            case_snakes[name] = snake
            case_histories[name] = history

            print(
                f"  {name:<12}  IoU={metrics['iou']:.3f}  "
                f"Haus={metrics['hausdorff']:.1f}  "
                f"MCD={metrics['mean_dist']:.2f}  "
                f"({elapsed:.1f}s)"
            )

        all_results[case_name] = case_results

        # ---- plots ----
        prefix = os.path.join(output_dir, case_name)

        fig = plot_comparison(
            image, case_snakes, gt_contour=gt_contour,
            title=case["name"],
            save_path=f"{prefix}_comparison.png",
        )
        plt.close(fig)

        fig = plot_all_methods_overlay(
            image, case_snakes, gt_contour=gt_contour,
            title=case["name"],
            save_path=f"{prefix}_overlay.png",
        )
        plt.close(fig)

        fig = plot_metrics_bar(case_results, save_path=f"{prefix}_metrics.png")
        plt.close(fig)

        fig = plot_convergence(
            image, case_histories, save_path=f"{prefix}_convergence.png"
        )
        plt.close(fig)

    # ---- summary table ----
    _print_summary(all_results, output_dir)
    return all_results


def _print_summary(all_results, output_dir):
    methods = ["Classical", "Adaptive", "Multiscale", "Combined"]
    header = f"{'Case':<28}" + "".join(f"{'  '+m:>14}" for m in methods)
    sep = "-" * len(header)

    lines = ["\n" + "=" * len(header), "  IoU Summary", "=" * len(header),
             header, sep]
    for case, res in all_results.items():
        row = f"{case:<28}" + "".join(f"{res[m]['iou']:>14.3f}" for m in methods)
        lines.append(row)
    lines += [sep, "\n  Hausdorff Distance Summary", sep, header, sep]
    for case, res in all_results.items():
        row = f"{case:<28}" + "".join(f"{res[m]['hausdorff']:>14.1f}" for m in methods)
        lines.append(row)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(text)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    run()

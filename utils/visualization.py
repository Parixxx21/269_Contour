import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


_METHOD_COLORS = {
    "Classical": "#2196F3",
    "Adaptive": "#4CAF50",
    "Multiscale": "#FF9800",
    "Combined": "#F44336",
}


def _closed(snake):
    """Append first point to close the contour for plotting."""
    return np.vstack([snake, snake[0]])


def plot_comparison(image, snakes_dict, gt_contour=None, title="", save_path=None):
    """
    Side-by-side comparison of snake results on one image.

    Parameters
    ----------
    image        : (H, W) ndarray
    snakes_dict  : {method_name: (N,2) snake array}
    gt_contour   : optional (M,2) ground-truth contour
    """
    ncols = len(snakes_dict) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    # Raw image
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    if gt_contour is not None:
        c = _closed(gt_contour)
        axes[0].plot(c[:, 1], c[:, 0], "w--", linewidth=1.5, label="GT")
    axes[0].set_title("Input + GT", fontsize=11)
    axes[0].axis("off")

    for idx, (name, snake) in enumerate(snakes_dict.items()):
        ax = axes[idx + 1]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        c = _closed(snake)
        color = _METHOD_COLORS.get(name, "red")
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=2)
        if gt_contour is not None:
            gc = _closed(gt_contour)
            ax.plot(gc[:, 1], gc[:, 0], "w--", linewidth=1, alpha=0.6)
        ax.set_title(name, fontsize=11, color=color)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence(image, histories_dict, save_path=None):
    """
    Show snake evolution history for each method.

    Parameters
    ----------
    histories_dict : {method_name: list of (N,2) snake arrays}
    """
    n = len(histories_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, history) in zip(axes, histories_dict.items()):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        color = _METHOD_COLORS.get(name, "red")
        for i, snake in enumerate(history):
            alpha = 0.2 + 0.8 * (i / max(len(history) - 1, 1))
            c = _closed(snake)
            ax.plot(c[:, 1], c[:, 0], color=color, alpha=alpha, linewidth=1)
        ax.set_title(f"{name}\n({len(history)} snapshots)", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_bar(results_dict, save_path=None):
    """
    Bar chart comparing IoU, Hausdorff distance, and mean contour distance
    across methods.

    Parameters
    ----------
    results_dict : {method_name: {metric_name: float}}
    """
    methods = list(results_dict.keys())
    metrics = ["iou", "hausdorff", "mean_dist"]
    labels = ["IoU ↑", "Hausdorff Distance ↓", "Mean Contour Dist ↓"]
    colors = [_METHOD_COLORS.get(m, "#999") for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, label in zip(axes, metrics, labels):
        vals = [results_dict[m][metric] for m in methods]
        bars = ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_all_methods_overlay(image, snakes_dict, gt_contour=None,
                             title="", save_path=None):
    """All methods overlaid on a single image for direct comparison."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)

    if gt_contour is not None:
        gc = _closed(gt_contour)
        ax.plot(gc[:, 1], gc[:, 0], "w--", linewidth=2, label="GT", zorder=5)

    for name, snake in snakes_dict.items():
        c = _closed(snake)
        color = _METHOD_COLORS.get(name, "red")
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=2, label=name)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

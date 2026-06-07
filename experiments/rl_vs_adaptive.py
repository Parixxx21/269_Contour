"""
RL v4 vs AdaptiveSnake comparison on synthetic test cases.

Usage
-----
    python experiments/rl_vs_adaptive.py
    python experiments/rl_vs_adaptive.py --model results/rl_ctrl/20260524_150224_v4/dqn_ctrl
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stable_baselines3 import DQN

from experiments.synthetic_images import generate_test_cases
from extensions.adaptive_rl.env import SnakeCtrlEnv, PRESETS, _fast_iou
from extensions.rl_snake_env import _circular_snake
from src.adaptive_snake import AdaptiveSnake
from src.full_adaptive_snake import FullAdaptiveSnake
from src.evaluation import evaluate_snake

DEFAULT_MODEL = "results/rl_ctrl/20260524_150224_v4/dqn_ctrl"
OUT_DIR       = "results/rl_vs_adaptive"

CASE_ORDER = ["clean_disk", "noisy_disk", "low_contrast_ellipse",
              "noisy_lc_ellipse", "noisy_star"]

CASE_LABELS = {
    "clean_disk":          "Clean Disk",
    "noisy_disk":          "Noisy Disk",
    "low_contrast_ellipse":"Low-Contrast Ellipse",
    "noisy_lc_ellipse":    "Noisy LC Ellipse",
    "noisy_star":          "Noisy Star",
}

ADAPTIVE_PARAMS = dict(
    alpha=0.015, beta_min=0.005, beta_max=0.3,
    k=5.0, gamma=5.0, sigma=8.0, n_iter=5000,
    update_every=20, wedge=1.0,
)

FULL_ADAPTIVE_PARAMS = dict(
    alpha=0.015, beta_min=0.005, beta_max=0.3,
    k=5.0, gamma_min=2.0, gamma_max=10.0, k_gamma=5.0,
    sigma=8.0, n_iter=5000, update_every=20, wedge=1.0,
)

N_REPEATS    = 1    # RL is deterministic — same result every time
RL_N_CHAINS  = 2   # chain N episodes end-to-end so RL also gets ~5000 total iters

COLORS = {
    "Adaptive":      "#4CAF50",
    "FullAdaptive":  "#FF9800",
    "RL v4":         "#E91E63",
}


def _run_rl_chained(model, env, n_chains=RL_N_CHAINS):
    """Run RL for n_chains episodes end-to-end; each starts where the last left off."""
    obs, _ = env.reset()
    done = truncated = False
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(int(action))

    for _ in range(n_chains - 1):
        prev_snake = env.get_snake().copy()
        obs, _ = env.reset()
        # Warm-start: override snake and its IoU baseline so reward is computed correctly
        env._snake    = prev_snake
        env._prev_iou = _fast_iou(prev_snake, env._gt_mask, env._image.shape)
        obs = env._observe()
        done = truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(int(action))


def run_rl(model, cases):
    """Run RL (chained) on every case. Returns dict case→list with one metrics dict."""
    results = {}
    for key, case in cases.items():
        env = SnakeCtrlEnv(cases={key: case})
        _run_rl_chained(model, env)
        results[key] = [env.get_metrics()]
    return results


def _run_snake_model(model, cases):
    """Generic runner: fits model on each case, returns metrics and final snakes."""
    results = {}
    snakes  = {}
    for key, case in cases.items():
        img  = case["image"]
        mask = case["mask"]
        gt_c = case["gt_contour"]
        init = _circular_snake(img.shape)
        final, _ = model.fit(img, init.copy())
        results[key] = evaluate_snake(final, mask, gt_c, img.shape)
        snakes[key]  = final
    return results, snakes


def run_adaptive(cases):
    return _run_snake_model(AdaptiveSnake(**ADAPTIVE_PARAMS), cases)


def run_full_adaptive(cases):
    return _run_snake_model(FullAdaptiveSnake(**FULL_ADAPTIVE_PARAMS), cases)


def run_rl_single(model, cases):
    """Return final snake for each case (for visual panels)."""
    snakes = {}
    for key, case in cases.items():
        env = SnakeCtrlEnv(cases={key: case})
        _run_rl_chained(model, env)
        snakes[key] = env.get_snake()
    return snakes


def plot_comparison(cases, rl_results, adap_results, fadap_results,
                    rl_snakes, adap_snakes, fadap_snakes):
    os.makedirs(OUT_DIR, exist_ok=True)
    n_cases = len(CASE_ORDER)

    # ── Figure 1: visual panels (one row per method, one column per case) ──────
    fig, axes = plt.subplots(3, n_cases, figsize=(4 * n_cases, 10))
    method_rows = [
        ("Adaptive",     adap_snakes,  adap_results,  None),
        ("FullAdaptive", fadap_snakes, fadap_results, None),
        ("RL v4",        rl_snakes,    rl_results,    rl_results),
    ]
    row_titles = ["Adaptive (β only)", "FullAdaptive (β + γ)", "RL v4 (β + temporal γ,σ)"]

    for row_idx, (mname, snk_dict, res_dict, rl_res) in enumerate(method_rows):
        for col_idx, key in enumerate(CASE_ORDER):
            ax  = axes[row_idx, col_idx]
            img = cases[key]["image"]
            gt_c = cases[key]["gt_contour"]
            ax.imshow(img, cmap="gray")
            ax.plot(gt_c[:, 1], gt_c[:, 0], "w--", lw=1.2, alpha=0.7)
            s = snk_dict[key]
            iou = (rl_res[key][0]["iou"] if rl_res else res_dict[key]["iou"])
            ax.plot(s[:, 1], s[:, 0], color=COLORS[mname], lw=2)
            ax.set_title(f"IoU={iou:.3f}", fontsize=9, color=COLORS[mname])
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(row_titles[row_idx], fontsize=9, fontweight="bold")

    for ax, key in zip(axes[0], CASE_ORDER):
        ax.set_title(f"{CASE_LABELS[key]}\nIoU={adap_results[key]['iou']:.3f}",
                     fontsize=9, color=COLORS["Adaptive"])

    fig.suptitle("Adaptive vs FullAdaptive vs RL v4", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path1 = os.path.join(OUT_DIR, "visual_comparison.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path1}")

    # ── Figure 2: metric bar charts (3 methods × 5 cases) ─────────────────────
    metrics    = ["iou", "hausdorff", "mean_dist"]
    m_labels   = ["IoU ↑", "Hausdorff ↓", "Mean Dist ↓"]
    x          = np.arange(n_cases)
    bar_w      = 0.25
    case_ticks = [CASE_LABELS[k] for k in CASE_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, mlabel in zip(axes, metrics, m_labels):
        adap_vals  = [adap_results[k][metric]  for k in CASE_ORDER]
        fadap_vals = [fadap_results[k][metric] for k in CASE_ORDER]
        rl_means   = [rl_results[k][0][metric] for k in CASE_ORDER]

        b1 = ax.bar(x - bar_w, adap_vals,  bar_w, color=COLORS["Adaptive"],
                    label="Adaptive",     alpha=0.85, edgecolor="white")
        b2 = ax.bar(x,          fadap_vals, bar_w, color=COLORS["FullAdaptive"],
                    label="FullAdaptive", alpha=0.85, edgecolor="white")
        b3 = ax.bar(x + bar_w,  rl_means,  bar_w, color=COLORS["RL v4"],
                    label="RL v4",        alpha=0.85, edgecolor="white")

        for bars, vals, col in [(b1, adap_vals, "Adaptive"),
                                (b2, fadap_vals, "FullAdaptive"),
                                (b3, rl_means,  "RL v4")]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(max(adap_vals), max(fadap_vals), max(rl_means)) * 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5,
                        color=COLORS[col])

        ax.set_xticks(x)
        ax.set_xticklabels(case_ticks, rotation=20, ha="right", fontsize=9)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Adaptive vs FullAdaptive vs RL v4 — Metrics (5000 iters)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path2 = os.path.join(OUT_DIR, "metric_comparison.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path2}")

    # ── Text table ────────────────────────────────────────────────────────────
    lines = ["Adaptive vs FullAdaptive vs RL v4  (5000 iters)\n",
             f"{'Case':<24}  {'Adap IoU':<10}  {'Full IoU':<10}  {'RL IoU':<10}  "
             f"{'Adap HD':<9}  {'Full HD':<9}  {'RL HD':<9}  "
             f"{'Adap MD':<9}  {'Full MD':<9}  {'RL MD'}"]
    lines.append("-" * 110)
    for k in CASE_ORDER:
        a  = adap_results[k]
        fa = fadap_results[k]
        rl = rl_results[k][0]
        lines.append(
            f"{CASE_LABELS[k]:<24}  "
            f"{a['iou']:.3f}       {fa['iou']:.3f}       {rl['iou']:.3f}      "
            f"{a['hausdorff']:.1f}       {fa['hausdorff']:.1f}       {rl['hausdorff']:.1f}      "
            f"{a['mean_dist']:.2f}      {fa['mean_dist']:.2f}      {rl['mean_dist']:.2f}"
        )
    tpath = os.path.join(OUT_DIR, "table.txt")
    with open(tpath, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nTable saved: {tpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    model_path = args.model
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]

    print(f"Loading RL model: {model_path}.zip")
    model = DQN.load(model_path)

    cases = generate_test_cases()
    # Filter to canonical order
    cases = {k: cases[k] for k in CASE_ORDER if k in cases}

    print("Running RL v4 (chained)...")
    rl_results = run_rl(model, cases)
    rl_snakes  = run_rl_single(model, cases)

    print("Running AdaptiveSnake...")
    adap_results, adap_snakes = run_adaptive(cases)

    print("Running FullAdaptiveSnake...")
    fadap_results, fadap_snakes = run_full_adaptive(cases)

    print("Plotting...")
    plot_comparison(cases, rl_results, adap_results, fadap_results,
                    rl_snakes, adap_snakes, fadap_snakes)


if __name__ == "__main__":
    main()

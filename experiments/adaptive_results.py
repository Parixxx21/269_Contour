"""
Adaptive snake family comparison:
  Classical | Adaptive | FullAdaptive | RL-Adaptive

Sections
--------
  1. Synthetic test cases (5 canonical shapes)
  2. Real data — Fluo-N2DL-HeLa
  3. Real data — Ultrasound Nerve Segmentation (from Kaggle, if downloaded)

Usage
-----
    python experiments/adaptive_results.py
    python experiments/adaptive_results.py --n-real 20
    python experiments/adaptive_results.py --skip-synth   # real data only
    python experiments/adaptive_results.py --skip-real    # synthetic only

"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

import tifffile
from skimage.measure import find_contours

from experiments.synthetic_images import generate_test_cases
from extensions.adaptive_rl.env import SnakeCtrlEnv, _fast_iou
from extensions.rl_snake_env import _circular_snake
from src.adaptive_snake import AdaptiveSnake
from src.full_adaptive_snake import FullAdaptiveSnake
from src.evaluation import evaluate_snake

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "results/rl_ctrl/20260524_150224_v4/dqn_ctrl"
OUT_DIR       = "results/adaptive_results"

METHODS = ["Classical", "Adaptive", "FullAdaptive", "RL-Adaptive"]

COLORS = {
    "Classical":    "#2196F3",
    "Adaptive":     "#4CAF50",
    "FullAdaptive": "#FF9800",
    "RL-Adaptive":  "#E91E63",
}

SYNTH_N_ITERS  = 5000
REAL_N_ITERS   = 5500
US_N_ITERS     = 8000   # ultrasound needs more iters (weaker edges)
RL_N_CHAINS    = 2      # chain 2 episodes → ~5000 iters total

CASE_ORDER = ["clean_disk", "noisy_disk", "low_contrast_ellipse",
              "noisy_lc_ellipse", "noisy_star"]
CASE_LABELS = {
    "clean_disk":           "Clean Disk",
    "noisy_disk":           "Noisy Disk",
    "low_contrast_ellipse": "Low-Contrast Ellipse",
    "noisy_lc_ellipse":     "Noisy LC Ellipse",
    "noisy_star":           "Noisy Star",
}


# ── Data loaders ───────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def contour_init(mask, n=120, scale=1.3):
    """Circular snake centred at mask centroid, radius = scale × RMS distance to centroid."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return _circular_snake(mask.shape, n=n)
    cy, cx = ys.mean(), xs.mean()
    r = scale * np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).mean())
    r = max(r, 20.0)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = np.clip(cy + r * np.sin(theta), 2, mask.shape[0] - 2)
    cols = np.clip(cx + r * np.cos(theta), 2, mask.shape[1] - 2)
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
                cell_mask = (seg == label)
                contours = find_contours(cell_mask.astype(float), 0.5)
                if not contours:
                    continue
                gt_c = contours[0]
                samples.append({
                    "image":      img_norm,
                    "mask":       cell_mask,
                    "gt_contour": gt_c,
                    "name":       f"HeLa_seq{seq}_frame{frame}_cell{label}",
                })
                if len(samples) >= n_images:
                    return samples
    print(f"  Loaded {len(samples)} Fluo-N2DL-HeLa samples")
    return samples


def load_ultrasound(n_images=20):
    """Load up to n_images nerve samples from Ultrasound Nerve Segmentation dataset."""
    base = os.path.join(_ROOT, "data", "ultrasound-nerve-segmentation", "train")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Ultrasound data not found: {base}")
    files = sorted(f for f in os.listdir(base)
                   if f.endswith(".tif") and "_mask" not in f)
    samples = []
    for fname in files:
        mask_path = os.path.join(base, fname.replace(".tif", "_mask.tif"))
        if not os.path.exists(mask_path):
            continue
        mask_raw = tifffile.imread(mask_path)
        if mask_raw.max() == 0:   # no nerve visible — skip
            continue
        img_raw = tifffile.imread(os.path.join(base, fname)).astype(np.float32)
        img_norm = (img_raw - img_raw.min()) / (img_raw.max() - img_raw.min() + 1e-8)
        cell_mask = mask_raw > 0
        contours = find_contours(cell_mask.astype(float), 0.5)
        if not contours:
            continue
        gt_c = max(contours, key=len)
        samples.append({
            "image":      img_norm,
            "mask":       cell_mask,
            "gt_contour": gt_c,
            "name":       fname.replace(".tif", ""),
        })
        if len(samples) >= n_images:
            break
    if not samples:
        raise FileNotFoundError(f"No valid ultrasound samples with non-empty masks in {base}")
    print(f"  Loaded {len(samples)} ultrasound samples")
    return samples


# ── Method builders ────────────────────────────────────────────────────────────

def build_classical(n_iter):
    # Classical = same force + same optimizer as Adaptive, but β is fixed globally (no spatial adaptation)
    # Using AdaptiveSnake with beta_min=beta_max gives constant β everywhere
    return AdaptiveSnake(alpha=0.015, beta_min=0.1, beta_max=0.1,
                         k=5.0, gamma=5.0, sigma=8.0, n_iter=n_iter,
                         update_every=n_iter + 1,  # matrix never recomputed (β is fixed)
                         wedge=1.0)

def build_adaptive(n_iter):
    return AdaptiveSnake(alpha=0.015, beta_min=0.005, beta_max=0.3,
                         k=5.0, gamma=5.0, sigma=8.0, n_iter=n_iter,
                         update_every=20, wedge=1.0)

def build_full_adaptive(n_iter):
    return FullAdaptiveSnake(alpha=0.015, beta_min=0.005, beta_max=0.3,
                             k=5.0, gamma_min=2.0, gamma_max=10.0, k_gamma=5.0,
                             sigma=8.0, n_iter=n_iter, update_every=20, wedge=1.0)


# ── RL helpers ─────────────────────────────────────────────────────────────────

def _rl_run_chained(model, env, init_snake=None):
    """Run RL_N_CHAINS episodes end-to-end; warm-start with init_snake if given."""
    obs, _ = env.reset()
    if init_snake is not None:
        env._snake    = init_snake.copy()
        env._prev_iou = _fast_iou(env._snake, env._gt_mask, env._image.shape)
        obs = env._observe()

    for chain in range(RL_N_CHAINS):
        done = truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(int(action))
        if chain < RL_N_CHAINS - 1:
            prev = env.get_snake().copy()
            obs, _ = env.reset()
            env._snake    = prev
            env._prev_iou = _fast_iou(prev, env._gt_mask, env._image.shape)
            obs = env._observe()


# ── Synthetic evaluation ───────────────────────────────────────────────────────

def run_synthetic(model):
    cases = generate_test_cases()
    cases = {k: cases[k] for k in CASE_ORDER if k in cases}

    results = {m: {} for m in METHODS}
    snakes  = {m: {} for m in METHODS}

    classical     = build_classical(SYNTH_N_ITERS)
    adaptive      = build_adaptive(SYNTH_N_ITERS)
    full_adaptive = build_full_adaptive(SYNTH_N_ITERS)

    for key, case in cases.items():
        img  = case["image"]
        mask = case["mask"]
        gt_c = case["gt_contour"]
        init = _circular_snake(img.shape)

        for mname, mdl in [("Classical",    classical),
                            ("Adaptive",     adaptive),
                            ("FullAdaptive", full_adaptive)]:
            s, _ = mdl.fit(img, init.copy())
            results[mname][key] = evaluate_snake(s, mask, gt_c, img.shape)
            snakes[mname][key]  = s

        env = SnakeCtrlEnv(cases={key: case})
        _rl_run_chained(model, env)
        results["RL-Adaptive"][key] = env.get_metrics()
        snakes["RL-Adaptive"][key]  = env.get_snake()

        print(f"  [synth] {key}  "
              + "  ".join(f"{m}={results[m][key]['iou']:.3f}" for m in METHODS))

    return cases, results, snakes


# ── Real data evaluation ───────────────────────────────────────────────────────

def run_real(model, samples, tag, n_iters=None):
    """Run all 4 methods on a list of pre-loaded samples. Returns (vis_samples, all_results)."""
    if not samples:
        print(f"  No samples for {tag}.")
        return [], {m: [] for m in METHODS}

    n_iters = n_iters or REAL_N_ITERS
    classical     = build_classical(n_iters)
    adaptive      = build_adaptive(n_iters)
    full_adaptive = build_full_adaptive(n_iters)

    all_results = {m: [] for m in METHODS}
    vis_samples = []

    for i, sample in enumerate(samples):
        img  = sample["image"]
        mask = sample["mask"]
        gt_c = sample["gt_contour"]
        name = sample["name"]
        init = contour_init(mask)

        row = {"image": img, "mask": mask, "gt_contour": gt_c,
               "name": name, "snakes": {}}

        for mname, mdl in [("Classical",    classical),
                            ("Adaptive",     adaptive),
                            ("FullAdaptive", full_adaptive)]:
            s, _ = mdl.fit(img, init.copy())
            m    = evaluate_snake(s, mask, gt_c, img.shape)
            all_results[mname].append(m)
            row["snakes"][mname] = s

        rl_case = {"image": img, "mask": mask, "gt_contour": gt_c}
        env = SnakeCtrlEnv(cases={"real": rl_case})
        _rl_run_chained(model, env, init_snake=init)
        m_rl = env.get_metrics()
        all_results["RL-Adaptive"].append(m_rl)
        row["snakes"]["RL-Adaptive"] = env.get_snake()

        if len(vis_samples) < 5:
            vis_samples.append(row)

        print(f"  [{tag} {i+1}/{len(samples)}] {name}  "
              + "  ".join(f"{mn}={all_results[mn][-1]['iou']:.3f}" for mn in METHODS))

    return vis_samples, all_results


# ── Plotting ───────────────────────────────────────────────────────────────────

def _metric_bars(ax, xlabels, method_vals, offset_scale=0.01):
    n_m   = len(METHODS)
    w     = 0.8 / n_m
    x     = np.arange(len(xlabels))
    all_v = [v for vals in method_vals.values() for v in vals]
    vrange = max(all_v) - min(all_v) + 1e-8

    for j, mname in enumerate(METHODS):
        vals = method_vals[mname]
        off  = (j - n_m / 2 + 0.5) * w
        bars = ax.bar(x + off, vals, w, color=COLORS[mname],
                      label=mname, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + vrange * offset_scale,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6, color=COLORS[mname])

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_synthetic(cases, results, snakes, out_dir):
    n_cases = len(CASE_ORDER)

    # Visual: methods × cases
    fig, axes = plt.subplots(len(METHODS), n_cases,
                             figsize=(3.5 * n_cases, 3.2 * len(METHODS)))
    for ci, key in enumerate(CASE_ORDER):
        img  = cases[key]["image"]
        gt_c = cases[key]["gt_contour"]
        axes[0, ci].set_title(f"{CASE_LABELS[key]}", fontsize=9, fontweight="bold")
        for ri, mname in enumerate(METHODS):
            ax  = axes[ri, ci]
            ax.imshow(img, cmap="gray")
            ax.plot(gt_c[:, 1], gt_c[:, 0], "w--", lw=1.2, alpha=0.6)
            s   = snakes[mname][key]
            iou = results[mname][key]["iou"]
            ax.plot(s[:, 1], s[:, 0], color=COLORS[mname], lw=1.8)
            ax.set_title(f"IoU={iou:.3f}", fontsize=8, color=COLORS[mname])
            ax.axis("off")
            if ci == 0:
                ax.set_ylabel(mname, fontsize=9, fontweight="bold",
                              color=COLORS[mname])

    fig.suptitle("Synthetic Benchmark — Classical / Adaptive / FullAdaptive / RL-Adaptive",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "synthetic_visual.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")

    # Metric bars
    xlabels = [CASE_LABELS[k] for k in CASE_ORDER]
    metrics_cfg = [("iou", "IoU ↑"), ("hausdorff", "Hausdorff ↓"), ("mean_dist", "Mean Dist ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, label) in zip(axes, metrics_cfg):
        vals = {m: [results[m][k][metric] for k in CASE_ORDER] for m in METHODS}
        _metric_bars(ax, xlabels, vals)
        ax.set_title(label, fontsize=11, fontweight="bold")
    fig.suptitle(f"Synthetic Metrics ({SYNTH_N_ITERS} iters)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "synthetic_metrics.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")

    # Table
    lines = [f"Synthetic benchmark  ({SYNTH_N_ITERS} iters)\n",
             f"{'Case':<24}" + "".join(f"  {m+' IoU':<16}" for m in METHODS)]
    lines.append("-" * 92)
    for k in CASE_ORDER:
        row = f"{CASE_LABELS[k]:<24}"
        for m in METHODS:
            row += f"  {results[m][k]['iou']:.3f}             "
        lines.append(row)
    p = os.path.join(out_dir, "synthetic_table.txt")
    with open(p, "w") as f: f.write("\n".join(lines))
    print("\n".join(lines)); print(f"Table: {p}")


def plot_real(vis_samples, all_results, tag, out_dir):
    n = len(all_results["Classical"])
    if n == 0:
        return

    # Visual: methods × up-to-5 samples
    n_vis = len(vis_samples)
    if n_vis > 0:
        fig, axes = plt.subplots(len(METHODS), n_vis,
                                 figsize=(3.5 * n_vis, 3.2 * len(METHODS)))
        if n_vis == 1:
            axes = axes[:, np.newaxis]
        for ci, samp in enumerate(vis_samples):
            img  = samp["image"]
            gt_c = samp["gt_contour"]
            short = samp["name"].split("_cell")[0][-12:]
            axes[0, ci].set_title(short, fontsize=8, fontweight="bold")
            for ri, mname in enumerate(METHODS):
                ax = axes[ri, ci]
                ax.imshow(img, cmap="gray")
                ax.plot(gt_c[:, 1], gt_c[:, 0], "w--", lw=1.2, alpha=0.6)
                s   = samp["snakes"][mname]
                iou = all_results[mname][ci]["iou"]
                ax.plot(s[:, 1], s[:, 0], color=COLORS[mname], lw=1.8)
                ax.set_title(f"IoU={iou:.3f}", fontsize=8, color=COLORS[mname])
                ax.axis("off")
                if ci == 0:
                    ax.set_ylabel(mname, fontsize=9, fontweight="bold",
                                  color=COLORS[mname])
        fig.suptitle(f"Real Data ({tag}) — Visual Comparison",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        p = os.path.join(out_dir, f"{tag}_visual.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {p}")

    # Mean ± std bar chart per method
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    metrics_cfg = [("iou", "IoU ↑"), ("hausdorff", "Hausdorff ↓"), ("mean_dist", "Mean Dist ↓")]
    x = np.arange(len(METHODS))
    for ax, (metric, label) in zip(axes, metrics_cfg):
        means  = [np.mean([r[metric] for r in all_results[m]]) for m in METHODS]
        stds   = [np.std( [r[metric] for r in all_results[m]]) for m in METHODS]
        colors = [COLORS[m] for m in METHODS]
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5,
                      alpha=0.85, edgecolor="white")
        for bar, v, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.15,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=15, ha="right", fontsize=9)
        ax.set_title(f"{label}  (n={n})", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Real Data ({tag}) — Mean ± Std  ({REAL_N_ITERS} iters)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_metrics.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p}")

    # Table
    lines = [f"Real data: {tag}  (n={n})\n",
             f"{'Method':<14}  {'IoU mean±std':<20}  {'Hausdorff':<20}  MeanDist"]
    lines.append("-" * 72)
    for m in METHODS:
        ious = [r["iou"]       for r in all_results[m]]
        hds  = [r["hausdorff"] for r in all_results[m]]
        mds  = [r["mean_dist"] for r in all_results[m]]
        lines.append(f"{m:<14}  {np.mean(ious):.3f}±{np.std(ious):.3f}            "
                     f"{np.mean(hds):.1f}±{np.std(hds):.1f}            "
                     f"{np.mean(mds):.2f}±{np.std(mds):.2f}")
    p = os.path.join(out_dir, f"{tag}_table.txt")
    with open(p, "w") as f: f.write("\n".join(lines))
    print("\n".join(lines)); print(f"Table: {p}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--n-real",     type=int, default=20)
    parser.add_argument("--skip-synth", action="store_true")
    parser.add_argument("--skip-real",  action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    model_path = args.model.replace(".zip", "")
    print(f"Loading RL model: {model_path}.zip")
    model = DQN.load(model_path)

    if not args.skip_synth:
        print("\n=== Synthetic benchmark ===")
        cases, syn_results, syn_snakes = run_synthetic(model)
        plot_synthetic(cases, syn_results, syn_snakes, OUT_DIR)

    if not args.skip_real:
        print(f"\n=== Real data: Fluo-HeLa (n={args.n_real}, {REAL_N_ITERS} iters) ===")
        fluo_samples = load_fluo_hela(n_images=args.n_real)
        vis, real_results = run_real(model, fluo_samples, "fluo", n_iters=REAL_N_ITERS)
        plot_real(vis, real_results, "fluo", OUT_DIR)

        print(f"\n=== Real data: Ultrasound (n={args.n_real}, {US_N_ITERS} iters) ===")
        try:
            us_samples = load_ultrasound(n_images=args.n_real)
            vis_us, us_results = run_real(model, us_samples, "ultrasound", n_iters=US_N_ITERS)
            plot_real(vis_us, us_results, "ultrasound", OUT_DIR)
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
            print("  → Download train.zip from kaggle.com/c/ultrasound-nerve-segmentation")
            print("    and extract to data/ultrasound-nerve-segmentation/train/")


if __name__ == "__main__":
    main()

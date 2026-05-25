"""
Train and evaluate the RL adaptive force controller (adjust γ and σ).

Usage
-----
    python extensions/adaptive_rl/train.py               # train
    python extensions/adaptive_rl/train.py --eval-only results/rl_ctrl/RUN_DIR
    python extensions/adaptive_rl/train.py --case noisy_star
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from extensions.adaptive_rl.env import SnakeCtrlEnv, PRESETS, N_ACTIONS
from experiments.synthetic_images import generate_test_cases

RESULTS_DIR  = "results/rl_ctrl"
TOTAL_STEPS  = 400_000


class ProgressLogger(BaseCallback):
    """Writes a one-line progress update to a file every LOG_EVERY steps."""
    LOG_EVERY = 5_000

    def __init__(self, log_path):
        super().__init__()
        self._log_path = log_path
        self._last_log = 0

    def _on_step(self):
        if self.num_timesteps - self._last_log >= self.LOG_EVERY:
            ep_rew = self.model.ep_info_buffer
            mean_rew = float(np.mean([e["r"] for e in ep_rew])) if ep_rew else 0.0
            line = (f"step={self.num_timesteps}  "
                    f"ep_rew={mean_rew:.4f}  "
                    f"eps={self.model.exploration_rate:.3f}\n")
            with open(self._log_path, "a") as f:
                f.write(line)
            self._last_log = self.num_timesteps
        return True


def _run_dir(tag=""):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    path = os.path.join(RESULTS_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def train(env, total_steps=TOTAL_STEPS, run_dir=None):
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        train_freq=4,
        target_update_interval=500,
        verbose=0,   # suppress stdout — progress tracked via progress.log
    )
    log_path = os.path.join(run_dir, "progress.log")
    model.learn(total_timesteps=total_steps,
                callback=ProgressLogger(log_path))
    path = os.path.join(run_dir, "dqn_ctrl")
    model.save(path)
    print(f"Model saved to {path}.zip")
    return model


def plot_episode(model, case_key="noisy_disk", run_dir=RESULTS_DIR):
    cases = generate_test_cases()
    env   = SnakeCtrlEnv(cases={case_key: cases[case_key]})
    obs, _ = env.reset()

    trajectory, rewards, presets_used = [env.get_snake().copy()], [], []
    done = truncated = False
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        action    = int(action)
        obs, r, done, truncated, _ = env.step(action)
        trajectory.append(env.get_snake().copy())
        rewards.append(r)
        if action < len(PRESETS):
            presets_used.append(action)

    image = cases[case_key]["image"]
    gt_c  = cases[case_key]["gt_contour"]
    m     = env.get_metrics()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Panel 1: init vs final
    axes[0].imshow(image, cmap="gray")
    axes[0].plot(trajectory[0][:, 1],  trajectory[0][:, 0], "b--", lw=1.5, label="Init")
    axes[0].plot(trajectory[-1][:, 1], trajectory[-1][:, 0], "r-",  lw=2,
                 label=f"RL ctrl (IoU={m.get('iou',0):.3f})")
    if gt_c is not None:
        axes[0].plot(gt_c[:, 1], gt_c[:, 0], "g-", lw=1.5, label="GT")
    axes[0].legend(fontsize=7); axes[0].axis("off")
    axes[0].set_title(f"RL Controller — {case_key}")

    # Panel 2: cumulative reward
    axes[1].plot(np.cumsum(rewards))
    axes[1].set_xlabel("RL step"); axes[1].set_ylabel("Cumulative IoU gain")
    axes[1].set_title("Episode reward")

    # Panel 3: preset usage histogram
    preset_labels = ["γ=1 σ=2", "γ=2 σ=3", "γ=5 σ=5", "γ=10 σ=6", "γ=20 σ=8"]
    counts = [presets_used.count(i) for i in range(len(PRESETS))]
    axes[2].bar(preset_labels, counts, color="steelblue")
    axes[2].set_title("Preset usage"); axes[2].tick_params(axis="x", rotation=30)

    # Panel 4: contour evolution
    axes[3].imshow(image, cmap="gray")
    step_size = max(1, len(trajectory) // 8)
    for i, s in enumerate(trajectory[::step_size]):
        alpha = 0.2 + 0.8 * (i / max(len(trajectory[::step_size]) - 1, 1))
        axes[3].plot(s[:, 1], s[:, 0], "r-", alpha=alpha, lw=1)
    if gt_c is not None:
        axes[3].plot(gt_c[:, 1], gt_c[:, 0], "g-", lw=1.5)
    axes[3].set_title("Contour evolution"); axes[3].axis("off")

    fig.tight_layout()
    out = os.path.join(run_dir, f"episode_{case_key}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out}")
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", metavar="RUN_DIR")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--case", default="noisy_disk")
    parser.add_argument("--tag",  default="")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.eval_only:
        model_path = args.eval_only
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "dqn_ctrl")
        model   = DQN.load(model_path)
        run_dir = os.path.dirname(model_path) \
                  if os.path.isfile(model_path + ".zip") else args.eval_only
        print("Loaded model from", model_path)
    else:
        run_dir = _run_dir(args.tag)
        print(f"Run directory: {run_dir}")
        with open(os.path.join(run_dir, "config.txt"), "w") as f:
            f.write(f"steps={args.steps}\ntag={args.tag}\n"
                    f"presets={PRESETS.tolist()}\n")
        # n_envs=1 avoids all multiprocessing issues on macOS
        env = make_vec_env(SnakeCtrlEnv, n_envs=1, vec_env_cls=DummyVecEnv)
        model = train(env, total_steps=args.steps, run_dir=run_dir)

    all_cases = ["noisy_disk", "noisy_star", "noisy_lc_ellipse",
                 "low_contrast_ellipse", "clean_disk"]
    eval_cases = [args.case] if args.case not in all_cases else [args.case]

    print("\n=== Evaluation ===")
    for case in eval_cases:
        m = plot_episode(model, case_key=case, run_dir=run_dir)
        print(f"  {case:<22} IoU={m.get('iou',0):.3f}  "
              f"Hausdorff={m.get('hausdorff',999):.1f}  "
              f"MeanDist={m.get('mean_dist',999):.2f}")


if __name__ == "__main__":
    main()

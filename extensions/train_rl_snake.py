"""
Train a DQN agent to control an active contour using stable-baselines3.

Usage
-----
    cd /Users/effiewu0628/Desktop/quarter3/269_project
    python extensions/train_rl_snake.py              # train + evaluate
    python extensions/train_rl_snake.py --eval-only  # load saved model and evaluate
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from extensions.rl_snake_env import SnakeEnv, CURV_LAMBDA
from extensions.rl_param_env import SnakeParamEnv
from experiments.synthetic_images import generate_test_cases

RESULTS_DIR = "results/rl"
TOTAL_STEPS = 200_000   # training budget (~5-15 min on CPU)


def _run_dir(tag=""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        learning_starts=2_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        train_freq=4,
        target_update_interval=500,
        verbose=1,
    )
    model.learn(total_timesteps=total_steps)
    model_path = os.path.join(run_dir, "dqn_snake")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    return model


def evaluate(model, n_episodes=20):
    """Run n_episodes and collect per-case IoU and Hausdorff scores."""
    cases = generate_test_cases()
    results = {k: {"iou": [], "hausdorff": []} for k in cases}

    for _ in range(n_episodes):
        env = SnakeEnv(cases=cases)
        obs, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(int(action))
        m = env.get_metrics()
        # identify which case was used (stored implicitly; re-derive from image)
        # For simplicity, record aggregate stats only
        for k in results:
            results[k]["iou"].append(m.get("iou", 0))
            results[k]["hausdorff"].append(m.get("hausdorff", 999))

    return results


def plot_episode(model, case_key="noisy_disk", run_dir=RESULTS_DIR, EnvClass=None):
    """Roll out one deterministic episode and save a figure."""
    if EnvClass is None:
        EnvClass = SnakeParamEnv
    cases = generate_test_cases()
    env = EnvClass(cases={case_key: cases[case_key]})
    obs, _ = env.reset()

    trajectory = [env.get_snake().copy()]
    rewards = []
    done = truncated = False
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, truncated, _ = env.step(int(action))
        trajectory.append(env.get_snake().copy())
        rewards.append(r)

    image = cases[case_key]["image"]
    gt_c  = cases[case_key]["gt_contour"]
    m     = env.get_metrics()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].plot(trajectory[0][:, 1],  trajectory[0][:, 0],
                 "b--", lw=1.5, label="Init")
    axes[0].plot(trajectory[-1][:, 1], trajectory[-1][:, 0],
                 "r-",  lw=2,   label=f"RL snake (IoU={m.get('iou',0):.3f})")
    if gt_c is not None:
        axes[0].plot(gt_c[:, 1], gt_c[:, 0], "g-", lw=1.5, label="GT")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"RL Snake — {case_key}")
    axes[0].axis("off")

    axes[1].plot(np.cumsum(rewards))
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative reward")
    axes[1].set_title("Episode reward")

    axes[2].imshow(image, cmap="gray")
    step_size = max(1, len(trajectory) // 10)
    for i, s in enumerate(trajectory[::step_size]):
        alpha = 0.3 + 0.7 * (i / max(len(trajectory[::step_size]) - 1, 1))
        axes[2].plot(s[:, 1], s[:, 0], "r-", alpha=alpha, lw=1)
    if gt_c is not None:
        axes[2].plot(gt_c[:, 1], gt_c[:, 0], "g-", lw=1.5)
    axes[2].set_title("Contour evolution")
    axes[2].axis("off")

    fig.tight_layout()
    out = os.path.join(run_dir, f"episode_{case_key}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Episode plot saved to {out}")
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", metavar="RUN_DIR",
                        help="path to a run dir (or model .zip) to load and evaluate")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--case", default="noisy_disk")
    parser.add_argument("--tag", default="",
                        help="short label appended to the timestamp run dir name")
    parser.add_argument("--env", default="param", choices=["param", "direct"],
                        help="param=RL controls snake parameters (new); "
                             "direct=RL controls contour points directly (old)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    EnvClass = SnakeParamEnv if args.env == "param" else SnakeEnv

    if args.eval_only:
        model_path = args.eval_only
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "dqn_snake")
        model = DQN.load(model_path)
        run_dir = os.path.dirname(model_path) if os.path.isfile(model_path + ".zip") \
                  else args.eval_only
        print("Loaded model from", model_path)
    else:
        run_dir = _run_dir(args.tag)
        print(f"Run directory: {run_dir}")
        with open(os.path.join(run_dir, "config.txt"), "w") as f:
            f.write(f"env={args.env}\nsteps={args.steps}\ntag={args.tag}\n")
        env = make_vec_env(EnvClass, n_envs=4, vec_env_cls=DummyVecEnv)
        model = train(env, total_steps=args.steps, run_dir=run_dir)

    print("\n=== Evaluation episode ===")
    m = plot_episode(model, case_key=args.case, run_dir=run_dir, EnvClass=EnvClass)
    print(f"  IoU={m.get('iou', 0):.3f}  "
          f"Hausdorff={m.get('hausdorff', 999):.1f}  "
          f"MeanDist={m.get('mean_dist', 999):.2f}")


if __name__ == "__main__":
    main()

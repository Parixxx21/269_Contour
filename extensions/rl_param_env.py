"""
RL environment: agent controls snake parameters (α, β, σ) as a learned scheduler.

Each RL action selects a (alpha, beta, sigma) preset, then the underlying
classical implicit-time-step snake runs ITERS_PER_STEP iterations with those
parameters.  The RL agent learns WHEN to use coarse vs fine parameters purely
from the current contour state — no manual schedule required.

Comparison story
----------------
  Classical  : fixed α/β/σ for all 2000 iterations
  Multiscale : hand-designed σ schedule (coarse→fine), fixed α/β
  Adaptive   : spatially-varying β per iteration, fixed α/σ
  RL (this)  : temporally-adaptive α/β/σ, learned from experience

Episode = MAX_STEPS (25) RL decisions × ITERS_PER_STEP (80) snake iters
        = 2000 total snake iterations  (same budget as classical methods)

Observation (104-dim)
---------------------
  [0:32]   gradient magnitude at 32 probe points
  [32:64]  sin of gradient direction at probe points
  [64:96]  cos of gradient direction at probe points
  [96:100] contour stats: mean_r, std_r, cy_off, cx_off
  [100]    episode progress  (step / MAX_STEPS)
  [101]    current alpha     (normalised to [0,1])
  [102]    current beta      (normalised to [0,1])
  [103]    current sigma     (normalised to [0,1])

Actions (7 discrete)
---------------------
  0  "coarse"     alpha=0.015  beta=0.50  sigma=8.0  — stiff+smooth, global pull
  1  "medium"     alpha=0.015  beta=0.10  sigma=5.0  — balanced
  2  "fine"       alpha=0.015  beta=0.05  sigma=3.0  — flexible, moderate detail
  3  "flexible"   alpha=0.005  beta=0.01  sigma=2.0  — loose+sharp, complex shapes
  4  "stiff"      alpha=0.050  beta=0.50  sigma=4.0  — resist deformation
  5  reparameterize (arc-length redistribution, no snake iters)
  6  done         — terminate episode early

Reward
------
  dense   : Δ(IoU) after each batch of snake iterations
  terminal: +1.0 if final IoU > 0.92
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.linalg import circulant

from src.evaluation import evaluate_snake
from extensions.rl_snake_env import (
    _random_case, _circular_snake,
    _gradient_features, _contour_stats,
)

# ── Constants ──────────────────────────────────────────────────────────────────
N_PROBES         = 32
N_POINTS         = 100
IMG_SHAPE        = (256, 256)
MAX_STEPS        = 25       # RL decisions per episode
ITERS_PER_STEP   = 80       # snake iters per RL action  (25×80 = 2000 total)
REPARAM_EVERY    = 20       # reparam inside each snake batch
GAMMA_SNAKE      = 5.0      # implicit time-step (fixed — only α/β/σ are learned)
IOU_BONUS_THRESHOLD = 0.92

# Parameter presets the agent chooses from
PARAM_PRESETS = [
    {"alpha": 0.015, "beta": 0.50, "sigma": 8.0},  # 0 coarse
    {"alpha": 0.015, "beta": 0.10, "sigma": 5.0},  # 1 medium
    {"alpha": 0.015, "beta": 0.05, "sigma": 3.0},  # 2 fine
    {"alpha": 0.005, "beta": 0.01, "sigma": 2.0},  # 3 flexible
    {"alpha": 0.050, "beta": 0.50, "sigma": 4.0},  # 4 stiff
]
N_PRESETS = len(PARAM_PRESETS)
N_ACTIONS = N_PRESETS + 2   # + reparameterize + done = 7

ALPHA_RANGE = (0.005, 0.050)
BETA_RANGE  = (0.010, 0.500)
SIGMA_RANGE = (2.0,   8.0)


# ── Snake helpers ──────────────────────────────────────────────────────────────

def _build_matrix(n, alpha, beta):
    row = np.zeros(n)
    row[0]  =  2*alpha + 6*beta
    row[1]  = -(alpha  + 4*beta)
    row[-1] = -(alpha  + 4*beta)
    row[2]  =  beta
    row[-2] =  beta
    return circulant(row)


def _reparameterize(snake):
    diffs = np.diff(snake, axis=0, prepend=snake[[-1]])
    arc   = np.cumsum(np.linalg.norm(diffs, axis=1))
    arc  -= arc[0]
    total = arc[-1]
    if total < 1e-8:
        return snake
    uniform   = np.linspace(0, total, len(snake), endpoint=False)
    arc_ext   = np.append(arc, total + arc[1])
    snake_ext = np.vstack([snake, snake[0]])
    new_y = np.interp(uniform, arc_ext, snake_ext[:, 0])
    new_x = np.interp(uniform, arc_ext, snake_ext[:, 1])
    return np.column_stack([new_y, new_x])


def _compute_force_field(image, sigma):
    img = gaussian_filter(image.astype(float), sigma=max(sigma, 0.5))
    gy, gx = np.gradient(img)
    fscale = max(np.abs(gx).max(), np.abs(gy).max(), 1e-8)
    return gy / fscale, gx / fscale   # (fy, fx)


# ── Environment ────────────────────────────────────────────────────────────────

class SnakeParamEnv(gym.Env):
    """
    RL controls snake parameters; the underlying solver handles shape fitting.
    Training: fresh random image every reset().
    Evaluation: pass cases=generate_test_cases() to fix the image set.
    """

    metadata = {"render_modes": []}

    def __init__(self, cases=None):
        super().__init__()
        self._fixed_cases = cases
        self._case_keys   = list(cases.keys()) if cases else None

        obs_dim = N_PROBES * 3 + 4 + 1 + 3   # 104
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._image      = None
        self._snake      = None
        self._gt_mask    = None
        self._gt_contour = None
        self._step_count = 0
        self._prev_iou   = 0.0
        self._cur_alpha  = PARAM_PRESETS[0]["alpha"]
        self._cur_beta   = PARAM_PRESETS[0]["beta"]
        self._cur_sigma  = PARAM_PRESETS[0]["sigma"]
        self._fy          = None
        self._fx          = None
        self._last_sigma  = None
        self._inv         = None
        self._last_alpha  = None
        self._last_beta   = None

    # ── Gym API ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._fixed_cases is not None:
            key  = self.np_random.choice(self._case_keys)
            case = self._fixed_cases[key]
            self._image      = case["image"].astype(np.float32)
            self._gt_mask    = case["mask"]
            self._gt_contour = case["gt_contour"]
            self._snake      = _circular_snake(self._image.shape)
        else:
            img, mask, gt_c, init = _random_case(self.np_random)
            self._image      = img
            self._gt_mask    = mask
            self._gt_contour = gt_c
            self._snake      = init

        self._step_count = 0
        self._cur_alpha  = PARAM_PRESETS[0]["alpha"]
        self._cur_beta   = PARAM_PRESETS[0]["beta"]
        self._cur_sigma  = PARAM_PRESETS[0]["sigma"]
        self._fy          = None
        self._fx          = None
        self._last_sigma  = None
        self._inv         = None
        self._last_alpha  = None
        self._last_beta   = None

        if self._gt_mask is not None:
            m = evaluate_snake(self._snake, self._gt_mask,
                               self._gt_contour, self._image.shape)
            self._prev_iou = float(m["iou"])
        else:
            self._prev_iou = 0.0

        return self._observe(), {}

    def step(self, action):
        done = False

        if action == N_ACTIONS - 1:           # done
            done = True
        elif action == N_ACTIONS - 2:         # reparameterize
            self._snake = _reparameterize(self._snake)
        else:                                 # preset → run snake
            p = PARAM_PRESETS[action]
            self._cur_alpha = p["alpha"]
            self._cur_beta  = p["beta"]
            self._cur_sigma = p["sigma"]
            self._snake = self._run_snake_iters(ITERS_PER_STEP)

        self._step_count += 1
        truncated = self._step_count >= MAX_STEPS

        if self._gt_mask is not None:
            m = evaluate_snake(self._snake, self._gt_mask,
                               self._gt_contour, self._image.shape)
            iou = float(m["iou"])
        else:
            iou = 0.0

        reward = iou - self._prev_iou
        self._prev_iou = iou

        if truncated and iou > IOU_BONUS_THRESHOLD:
            reward += 1.0

        return self._observe(), float(reward), done, truncated, {}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_snake_iters(self, n_iters):
        snake = self._snake.copy()
        h, w  = self._image.shape
        n     = len(snake)

        # Recompute force field only when sigma changes
        if self._last_sigma != self._cur_sigma:
            self._fy, self._fx = _compute_force_field(self._image, self._cur_sigma)
            self._last_sigma   = self._cur_sigma

        # Recompute matrix inverse only when alpha/beta change
        if (self._last_alpha != self._cur_alpha or
                self._last_beta != self._cur_beta or self._inv is None):
            A = _build_matrix(n, self._cur_alpha, self._cur_beta)
            self._inv        = np.linalg.inv(A + GAMMA_SNAKE * np.eye(n))
            self._last_alpha = self._cur_alpha
            self._last_beta  = self._cur_beta
        inv = self._inv

        for it in range(n_iters):
            y = np.clip(snake[:, 0], 0, h - 1)
            x = np.clip(snake[:, 1], 0, w - 1)
            fyn = map_coordinates(self._fy, [y, x], order=1, mode="nearest")
            fxn = map_coordinates(self._fx, [y, x], order=1, mode="nearest")

            yn = inv @ (GAMMA_SNAKE * snake[:, 0] + fyn)
            xn = inv @ (GAMMA_SNAKE * snake[:, 1] + fxn)
            snake[:, 0] = np.clip(yn, 0, h - 1)
            snake[:, 1] = np.clip(xn, 0, w - 1)

            if (it + 1) % REPARAM_EVERY == 0:
                snake = _reparameterize(snake)

        return snake

    def _observe(self):
        img_smooth = gaussian_filter(self._image, sigma=max(self._cur_sigma, 0.5))
        mag, sdir, cdir = _gradient_features(img_smooth, self._snake, N_PROBES)
        stats    = _contour_stats(self._snake, self._image.shape)
        progress = np.array([self._step_count / MAX_STEPS], dtype=np.float32)

        def _norm(v, lo, hi):
            return np.array([(v - lo) / (hi - lo + 1e-8)], dtype=np.float32)

        return np.concatenate([
            mag.astype(np.float32),
            sdir.astype(np.float32),
            cdir.astype(np.float32),
            stats,
            progress,
            _norm(self._cur_alpha, *ALPHA_RANGE),
            _norm(self._cur_beta,  *BETA_RANGE),
            _norm(self._cur_sigma, *SIGMA_RANGE),
        ])

    def get_snake(self):
        return self._snake.copy()

    def get_metrics(self):
        if self._gt_mask is None:
            return {}
        return evaluate_snake(self._snake, self._gt_mask,
                              self._gt_contour, self._image.shape)

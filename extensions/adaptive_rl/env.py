"""
RL Adaptive Force Controller — wraps AdaptiveSnake logic.

The RL agent learns WHEN to use which (γ, σ) combination, adding
temporal adaptivity on top of AdaptiveSnake's spatial adaptivity (per-point β).

AdaptiveSnake already handles:
  - spatially-varying β_i: near strong edges β is small (snake deforms freely),
    in flat/noisy areas β is large (strong regularisation)

RL adds:
  - temporally-varying γ and σ: agent decides each 50-iter chunk whether to
    be aggressive (low γ, fine σ) or conservative (high γ, coarse σ)

Update rule (same as AdaptiveSnake):
    (A(β) + γI) v^{t+1} = γ v^t + F_ext(v^t)

where A(β) is the per-point stiffness matrix and
      F_ext = -∇(-wedge·|∇(G_σ*I)|²)  (edge-energy gradient, pulls toward edges)

Episode budget: MAX_STEPS (40) × ITERS_PER_STEP (50) = 2000 snake iters
               (same total as classical methods in main.py)

Observation (104-dim)
---------------------
  [0:32]   gradient magnitude at 32 probe points (at current σ)
  [32:64]  sin of gradient angle at probe points
  [64:96]  cos of gradient angle at probe points
  [96:100] contour stats: mean_r, std_r, cy_off, cx_off
  [100]    episode progress (step / MAX_STEPS)
  [101]    current γ normalised to [0,1]
  [102]    current σ normalised to [0,1]
  [103]    mean gradient magnitude along contour (edge strength indicator)

Actions (8 discrete)
---------------------
  0  γ=1,  σ=2  — aggressive: full trust in force, fine edges
  1  γ=2,  σ=3  — confident
  2  γ=5,  σ=5  — balanced (AdaptiveSnake default)
  3  γ=10, σ=6  — cautious (AdaptiveSnake default σ)
  4  γ=20, σ=8  — conservative: near-frozen, coarse edges only
  5  reparameterize — arc-length redistribution, no snake iters
  6  done        — terminate episode early

Reward
------
  dense   : Δ(IoU) after each batch of ITERS_PER_STEP snake iterations
  terminal: +1.0 if final IoU > 0.92 at episode end (timeout only)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import map_coordinates, gaussian_filter

from skimage.draw import polygon as sk_polygon
from src.evaluation import evaluate_snake
from src.energy import compute_external_forces, compute_gradient_magnitude
from extensions.rl_snake_env import (
    _random_case, _circular_snake, _gradient_features, _contour_stats,
)

# Consts
N_PROBES          = 32
N_POINTS          = 100
MAX_STEPS         = 50        # RL decisions per episode
ITERS_PER_STEP    = 50        # snake iters per RL action  (50×50 = 2500 total)
UPDATE_EVERY      = 20        # recompute adaptive β matrix every N iters
REPARAM_EVERY     = 50        # arc-length reparam inside each batch
ALPHA_SNAKE       = 0.015     # elasticity — fixed
BETA_MIN          = 0.005     # adaptive β lower bound (matches AdaptiveSnake)
BETA_MAX          = 0.3       # adaptive β upper bound (matches AdaptiveSnake)
K_SENSITIVITY     = 5.0       # how fast β drops near strong edges (matches AdaptiveSnake)
IOU_BONUS_THRESHOLD = 0.92

# (γ, σ) presets: rows of [gamma, sigma]
# γ controls force vs smoothness trust (low γ = aggressive, high γ = conservative)
# σ controls edge scale (low σ = fine detail, high σ = noise-robust)
PRESETS = np.array([
    [1.0,  2.0],   # 0 aggressive: full trust in force, fine edges
    [2.0,  3.0],   # 1 confident
    [5.0,  5.0],   # 2 balanced  (AdaptiveSnake default γ=5)
    [10.0, 6.0],   # 3 cautious  (AdaptiveSnake default σ=6)
    [20.0, 8.0],   # 4 conservative: near-frozen, coarse edges only
], dtype=np.float64)

N_PRESETS  = len(PRESETS)
N_ACTIONS  = N_PRESETS + 2    # + reparameterize + done = 7

GAMMA_RANGE = (1.0,  20.0)
SIGMA_RANGE = (2.0,   8.0)


#  Fast IoU (no Hausdorff) — it is used during training for speed 

def _fast_iou(snake, gt_mask, shape):
    """Rasterise snake polygon and compute pixel-wise IoU. ~0.3ms vs 3.8ms."""
    rr, cc = sk_polygon(
        np.clip(snake[:, 0], 0, shape[0] - 1),
        np.clip(snake[:, 1], 0, shape[1] - 1),
        shape=shape,
    )
    pred = np.zeros(shape, dtype=bool)
    pred[rr, cc] = True
    inter = (pred & gt_mask).sum()
    union = (pred | gt_mask).sum()
    return float(inter) / (float(union) + 1e-8)


# Adaptive Snake helpers 

def _adaptive_beta(image, snake, sigma, k=K_SENSITIVITY,
                   beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Per-point β from local gradient magnitude (AdaptiveSnake formula)."""
    gmag = compute_gradient_magnitude(image, sigma=sigma)
    h, w = gmag.shape
    y = np.clip(snake[:, 0], 0, h - 1)
    x = np.clip(snake[:, 1], 0, w - 1)
    local_g = map_coordinates(gmag, [y, x], order=1, mode="nearest")
    beta = beta_min + (beta_max - beta_min) * np.exp(-k * local_g)
    return np.clip(beta, beta_min, beta_max)


def _build_adaptive_matrix(n, beta_array, alpha=ALPHA_SNAKE):
    """Per-point pentadiagonal stiffness matrix (matches AdaptiveSnake)."""
    A = np.zeros((n, n))
    a = alpha
    for i in range(n):
        b = beta_array[i]
        A[i,  i]         = 2 * a + 6 * b
        A[i, (i+1) % n]  = -(a + 4 * b)
        A[i, (i-1) % n]  = -(a + 4 * b)
        A[i, (i+2) % n]  = b
        A[i, (i-2) % n]  = b
    return A


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


# Environment controls

class SnakeCtrlEnv(gym.Env):
    """
    RL controls γ (force trust) and σ (edge scale) at each step.
    Underlying snake uses AdaptiveSnake's update rule with per-point β.
    """

    metadata = {"render_modes": []}

    def __init__(self, cases=None):
        super().__init__()
        self._fixed_cases = cases
        self._case_keys   = list(cases.keys()) if cases else None

        obs_dim = N_PROBES * 3 + 4 + 1 + 2 + 1   # 104
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
        self._cur_gamma  = PRESETS[2, 0]
        self._cur_sigma  = PRESETS[2, 1]

        # Cached force field (invalidated when σ changes)
        self._fy         = None
        self._fx         = None
        self._last_sigma = None

        # Cached smoothed image for observation features
        self._img_smooth  = None

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
        self._cur_gamma  = PRESETS[2, 0]   # start at balanced preset
        self._cur_sigma  = PRESETS[2, 1]
        self._fy         = None
        self._fx         = None
        self._last_sigma = None
        self._img_smooth = None

        if self._gt_mask is not None:
            self._prev_iou = _fast_iou(self._snake, self._gt_mask, self._image.shape)
        else:
            self._prev_iou = 0.0

        return self._observe(), {}

    def step(self, action):
        done = False

        if action == N_ACTIONS - 1:           # done
            done = True
        elif action == N_ACTIONS - 2:         # reparameterize only
            self._snake = _reparameterize(self._snake)
        else:                                 # select preset → run snake
            self._cur_gamma = PRESETS[action, 0]
            self._cur_sigma = PRESETS[action, 1]
            self._snake = self._run_snake_iters(ITERS_PER_STEP)

        self._step_count += 1
        truncated = self._step_count >= MAX_STEPS

        if self._gt_mask is not None:
            iou = _fast_iou(self._snake, self._gt_mask, self._image.shape)
        else:
            iou = 0.0

        reward = iou - self._prev_iou
        self._prev_iou = iou

        if truncated and iou > IOU_BONUS_THRESHOLD:
            reward += 1.0

        return self._observe(), float(reward), done, truncated, {}


    def _run_snake_iters(self, n_iters):
        """
        Run AdaptiveSnake update rule for n_iters with current γ and σ.
        - Force field: compute_external_forces (edge-energy gradient) — same as AdaptiveSnake
        - Matrix: per-point adaptive β — same as AdaptiveSnake
        - RL controls γ and σ (temporal); β handles spatial adaptation internally
        """
        snake = self._snake.copy()
        h, w  = self._image.shape
        gamma = self._cur_gamma

        # Recompute force field only when σ changes
        if self._last_sigma != self._cur_sigma:
            fy, fx = compute_external_forces(
                self._image, sigma=self._cur_sigma, wline=0.0, wedge=1.0
            )
            fscale = max(np.abs(fx).max(), np.abs(fy).max(), 1e-8)
            self._fy = fy / fscale
            self._fx = fx / fscale
            self._last_sigma = self._cur_sigma
            self._img_smooth = None   # observation cache also invalidated

        inv = None

        for it in range(n_iters):
            # Recompute adaptive β matrix every UPDATE_EVERY iters
            if it % UPDATE_EVERY == 0:
                beta_arr = _adaptive_beta(self._image, snake, self._cur_sigma)
                A   = _build_adaptive_matrix(len(snake), beta_arr)
                inv = np.linalg.inv(A + gamma * np.eye(len(snake)))

            y = np.clip(snake[:, 0], 0, h - 1)
            x = np.clip(snake[:, 1], 0, w - 1)
            fyn = map_coordinates(self._fy, [y, x], order=1, mode="nearest")
            fxn = map_coordinates(self._fx, [y, x], order=1, mode="nearest")

            snake[:, 0] = np.clip(inv @ (gamma * snake[:, 0] + fyn), 0, h - 1)
            snake[:, 1] = np.clip(inv @ (gamma * snake[:, 1] + fxn), 0, w - 1)

            if (it + 1) % REPARAM_EVERY == 0:
                snake = _reparameterize(snake)

        return snake

    def _observe(self):
        if self._img_smooth is None:
            self._img_smooth = gaussian_filter(
                self._image.astype(float), sigma=max(self._cur_sigma, 0.5)
            )
        mag, sdir, cdir = _gradient_features(self._img_smooth, self._snake, N_PROBES)
        stats    = _contour_stats(self._snake, self._image.shape)
        progress = np.array([self._step_count / MAX_STEPS], dtype=np.float32)

        def _norm(v, lo, hi):
            return np.array([(v - lo) / (hi - lo + 1e-8)], dtype=np.float32)

        mean_grad = np.array([float(mag.mean())], dtype=np.float32)

        return np.concatenate([
            mag.astype(np.float32),
            sdir.astype(np.float32),
            cdir.astype(np.float32),
            stats,
            progress,
            _norm(self._cur_gamma, *GAMMA_RANGE),
            _norm(self._cur_sigma, *SIGMA_RANGE),
            mean_grad,
        ])

    def get_snake(self):
        return self._snake.copy()

    def get_metrics(self):
        if self._gt_mask is None:
            return {}
        return evaluate_snake(self._snake, self._gt_mask,
                              self._gt_contour, self._image.shape)

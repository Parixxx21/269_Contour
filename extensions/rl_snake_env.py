"""
Gymnasium environment for RL-based active contour (snake) optimization.

Training uses procedurally generated random images (new shape every reset),
so the agent must learn a generalizable policy rather than memorize trajectories.
The original 5 synthetic test cases are reserved for evaluation only.

Two-phase episode design
------------------------
Phase 1 (steps 0 – PHASE_SWITCH-1): global actions only.
    Agent learns coarse positioning: get the contour roughly onto the boundary.

Phase 2 (steps PHASE_SWITCH – MAX_STEPS-1): global + local arc actions.
    Agent can now also deform individual quadrant arcs to fit non-circular shapes.
    Local actions are no-ops in Phase 1, so DQN learns to ignore them early,
    then discovers them once Phase 2 unlocks.

Observation (101-dim vector)
----------------------------
  [0:32]   gradient magnitude at 32 evenly-spaced contour probe points  [0,1]
  [32:64]  sin of gradient angle at each probe point  [-1,1]
  [64:96]  cos of gradient angle at each probe point  [-1,1]
  [96]     mean radius (normalised by image size)
  [97]     std of radius (how non-circular the contour is)
  [98]     centroid row offset from image centre (normalised)
  [99]     centroid col offset from image centre (normalised)
  [100]    current phase (0.0 = Phase 1, 1.0 = Phase 2)

Actions (discrete, 15)
----------------------
  Global (always active):
    0  inflate   — expand all points radially outward by STEP px
    1  deflate   — contract all points radially inward by STEP px
    2  shift up
    3  shift down
    4  shift left
    5  shift right
    6  done      — terminate episode early (no bonus, purely a stop signal)

  Local arcs (Phase 2 only — no-op in Phase 1):
    7   expand top arc     (points with angle in [-135°, -45°] from centroid)
    8   expand right arc   (angle in [-45°,  +45°])
    9   expand bottom arc  (angle in [+45°, +135°])
    10  expand left arc    (angle in [+135°, +225°] / [-180°, -135°])
    11  contract top arc
    12  contract right arc
    13  contract bottom arc
    14  contract left arc

Reward
------
  dense   : Δ(IoU) each step — clean signal, no constant negative drag
  terminal: +1.0 bonus if final IoU > IOU_BONUS_THRESHOLD (0.92)
            − CURV_LAMBDA * mean_curvature penalty at episode end only
            (curvature kept out of per-step reward to avoid drowning Δ IoU)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage.measure import find_contours

from src.evaluation import evaluate_snake
from utils.image_utils import add_gaussian_noise, reduce_contrast

# ── Constants ─────────────────────────────────────────────────────────────────
N_PROBES     = 32    # gradient samples along contour
STEP         = 3     # pixels per action
SIGMA        = 2.0   # gradient smoothing
MAX_STEPS    = 200   # max actions per episode
PHASE_SWITCH = 100   # step at which local arc actions unlock
N_GLOBAL     = 7     # number of global actions (incl. done)
N_LOCAL      = 8     # number of local arc actions (4 sectors × expand/contract)
N_ACTIONS    = N_GLOBAL + N_LOCAL   # 15 total
IOU_BONUS_THRESHOLD = 0.92   # raised from 0.85 — agent must fit well to earn bonus
N_POINTS     = 100   # snake resolution
IMG_SHAPE    = (256, 256)
CURV_LAMBDA  = 0.01  # curvature penalty weight in reward (mild regulariser)

# ── Random shape generators ───────────────────────────────────────────────────

def _disk_mask(shape, center, radius):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return (y - center[0]) ** 2 + (x - center[1]) ** 2 < radius ** 2


def _ellipse_mask(shape, center, a, b, angle):
    y, x = np.mgrid[:shape[0], :shape[1]]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr =  cos_a * (x - center[1]) + sin_a * (y - center[0])
    yr = -sin_a * (x - center[1]) + cos_a * (y - center[0])
    return (xr / a) ** 2 + (yr / b) ** 2 < 1


def _star_mask(shape, center, r_outer, r_inner, n_points):
    y, x = np.mgrid[:shape[0], :shape[1]]
    theta = np.arctan2(y - center[0], x - center[1])
    dist  = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    r_star = r_inner + (r_outer - r_inner) * (np.cos(n_points * theta) + 1) / 2
    return dist < r_star


def _polygon_mask(shape, center, n_verts, r_mean, r_jitter, rng):
    """Convex-ish random polygon via radial jitter."""
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_verts))
    radii  = r_mean + rng.uniform(-r_jitter, r_jitter, n_verts)
    verts_y = center[0] + radii * np.sin(angles)
    verts_x = center[1] + radii * np.cos(angles)
    # Rasterise with skimage draw
    from skimage.draw import polygon as sk_polygon
    mask = np.zeros(shape, dtype=bool)
    rr, cc = sk_polygon(verts_y, verts_x, shape=shape)
    mask[rr, cc] = True
    return mask


def _get_gt_contour(mask):
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    return max(contours, key=len)


def _random_case(rng, shape=IMG_SHAPE):
    """
    Generate one random training case.
    Returns (image, mask, gt_contour, init_snake).
    """
    margin = 50   # keep shapes away from borders so init circle is inside image
    cy = int(rng.integers(margin + 20, shape[0] - margin - 20))
    cx = int(rng.integers(margin + 20, shape[1] - margin - 20))

    kind = rng.choice(['disk', 'ellipse', 'star', 'polygon'])

    if kind == 'disk':
        r = int(rng.integers(25, 65))
        mask = _disk_mask(shape, (cy, cx), r)

    elif kind == 'ellipse':
        a = int(rng.integers(35, 70))
        b = int(rng.integers(20, a))
        angle = rng.uniform(0, np.pi)
        mask = _ellipse_mask(shape, (cy, cx), a, b, angle)

    elif kind == 'star':
        r_outer = int(rng.integers(35, 65))
        r_inner = int(rng.integers(15, max(16, r_outer - 10)))
        n_pts   = int(rng.integers(4, 8))
        mask = _star_mask(shape, (cy, cx), r_outer, r_inner, n_pts)

    else:  # polygon
        n_verts = int(rng.integers(5, 10))
        r_mean  = int(rng.integers(30, 60))
        r_jitter = int(rng.integers(5, 15))
        mask = _polygon_mask(shape, (cy, cx), n_verts, r_mean, r_jitter, rng)

    # Skip degenerate masks
    if mask.sum() < 200:
        return _random_case(rng, shape)

    # Random appearance
    img = mask.astype(float)
    contrast = rng.uniform(0.3, 1.0)
    img = reduce_contrast(img, factor=float(contrast))
    noise_sigma = rng.uniform(0.0, 0.20)
    if noise_sigma > 0.01:
        img = add_gaussian_noise(img, sigma=float(noise_sigma),
                                 seed=int(rng.integers(0, 1_000_000)))

    gt_contour = _get_gt_contour(mask)

    # Init snake: circle centred near (but not exactly at) shape centroid,
    # radius slightly larger than shape's bounding radius so it starts outside.
    ys, xs = np.where(mask)
    shape_r = max(
        np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).max() + 5,
        30.0
    )
    init_r   = min(shape_r + int(rng.integers(5, 20)), min(shape) // 2 - 5)
    init_off_y = int(rng.integers(-8, 9))
    init_off_x = int(rng.integers(-8, 9))
    init_snake = _circular_snake(
        shape,
        center=(cy + init_off_y, cx + init_off_x),
        radius=float(init_r),
    )

    return img.astype(np.float32), mask, gt_contour, init_snake


def _circular_snake(shape, center=None, radius=80.0, n=N_POINTS):
    if center is None:
        center = (shape[0] // 2, shape[1] // 2)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows  = center[0] + radius * np.sin(theta)
    cols  = center[1] + radius * np.cos(theta)
    return np.column_stack([rows, cols])


# ── Observation helpers ───────────────────────────────────────────────────────

def _sample_at_probes(field, snake, n_probes, h, w):
    idx = np.round(np.linspace(0, len(snake) - 1, n_probes,
                               endpoint=False)).astype(int)
    pts = snake[idx]
    y = np.clip(pts[:, 0], 0, h - 1)
    x = np.clip(pts[:, 1], 0, w - 1)
    return map_coordinates(field, [y, x], order=1, mode="nearest")


def _gradient_features(image_smooth, snake, n_probes=N_PROBES):
    """Return (mag, sin_dir, cos_dir) each of length n_probes."""
    h, w = image_smooth.shape
    gy, gx = np.gradient(image_smooth)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    denom = gmag.max()
    if denom > 0:
        gmag_norm = gmag / denom
    else:
        gmag_norm = gmag
    eps = 1e-8
    sin_dir = gy / (gmag + eps)
    cos_dir = gx / (gmag + eps)
    mag  = _sample_at_probes(gmag_norm, snake, n_probes, h, w)
    sdir = _sample_at_probes(sin_dir,   snake, n_probes, h, w)
    cdir = _sample_at_probes(cos_dir,   snake, n_probes, h, w)
    return mag, sdir, cdir


def _gradient_at_contour(image_smooth, snake, n_probes=N_PROBES):
    """Return only magnitude (used for backward-compat)."""
    mag, _, _ = _gradient_features(image_smooth, snake, n_probes)
    return mag


def _contour_curvature(snake):
    """Mean second-difference magnitude — proxy for total curvature."""
    prev_ = np.roll(snake,  1, axis=0)
    next_ = np.roll(snake, -1, axis=0)
    return np.linalg.norm(prev_ - 2 * snake + next_, axis=1).mean()


def _contour_stats(snake, image_shape):
    cy, cx = snake[:, 0].mean(), snake[:, 1].mean()
    ic, jc = image_shape[0] / 2, image_shape[1] / 2
    dists  = np.linalg.norm(snake - np.array([cy, cx]), axis=1)
    mean_r = dists.mean() / max(image_shape)
    std_r  = dists.std()  / max(image_shape)
    cy_off = (cy - ic) / image_shape[0]
    cx_off = (cx - jc) / image_shape[1]
    return np.array([mean_r, std_r, cy_off, cx_off], dtype=np.float32)


# ── Environment ───────────────────────────────────────────────────────────────

class SnakeEnv(gym.Env):
    """
    Training: new random image every reset() — agent must generalise.
    Evaluation: pass cases=generate_test_cases() to fix the image set.
    """

    metadata = {"render_modes": []}

    def __init__(self, cases=None, sigma=SIGMA):
        super().__init__()
        # If cases dict is provided, use fixed set (eval mode).
        # Otherwise, generate a fresh random image each episode (train mode).
        self._fixed_cases  = cases
        self._case_keys    = list(cases.keys()) if cases else None
        self.sigma         = sigma

        obs_dim = N_PROBES * 3 + 4 + 1  # mag + sin + cos + stats + phase = 101
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._image        = None
        self._image_smooth = None
        self._snake        = None
        self._gt_mask      = None
        self._gt_contour   = None
        self._step_count   = 0
        self._prev_iou     = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._fixed_cases is not None:
            # Evaluation mode: cycle through fixed cases
            key  = self.np_random.choice(self._case_keys)
            case = self._fixed_cases[key]
            self._image      = case["image"].astype(np.float32)
            self._gt_mask    = case["mask"]
            self._gt_contour = case["gt_contour"]
            self._snake      = _circular_snake(self._image.shape)
        else:
            # Training mode: new random scene every episode
            img, mask, gt_c, init = _random_case(self.np_random)
            self._image      = img
            self._gt_mask    = mask
            self._gt_contour = gt_c
            self._snake      = init

        self._image_smooth = gaussian_filter(self._image, sigma=self.sigma)
        self._step_count   = 0
        if self._gt_mask is not None:
            m = evaluate_snake(self._snake, self._gt_mask,
                               self._gt_contour, self._image.shape)
            self._prev_iou = float(m["iou"])
        else:
            self._prev_iou = 0.0

        return self._observe(), {}

    def step(self, action):
        h, w    = self._image.shape
        snake   = self._snake.copy()
        done    = False
        phase2  = self._step_count >= PHASE_SWITCH

        if action == 6:
            done = True

        elif action < N_GLOBAL:
            # ── Global actions (always active) ──────────────────────────────
            cy = snake[:, 0].mean()
            cx = snake[:, 1].mean()
            if action == 0:
                dirs  = snake - np.array([cy, cx])
                norms = np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-8)
                snake += STEP * dirs / norms
            elif action == 1:
                dirs  = snake - np.array([cy, cx])
                norms = np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-8)
                snake -= STEP * dirs / norms
            elif action == 2: snake[:, 0] -= STEP
            elif action == 3: snake[:, 0] += STEP
            elif action == 4: snake[:, 1] -= STEP
            elif action == 5: snake[:, 1] += STEP

        elif phase2:
            # ── Local arc actions (Phase 2 only) ────────────────────────────
            # Actions 7–10: expand top/right/bottom/left arc
            # Actions 11–14: contract top/right/bottom/left arc
            local_idx = action - N_GLOBAL          # 0–7
            sector    = local_idx % 4              # 0=top 1=right 2=bottom 3=left
            expand    = local_idx < 4

            cy = snake[:, 0].mean()
            cx = snake[:, 1].mean()
            angles = np.arctan2(snake[:, 0] - cy, snake[:, 1] - cx)  # [-π, π]

            # Sector angle ranges (image coords: row↓, col→)
            sector_masks = [
                (angles >= -3*np.pi/4) & (angles < -np.pi/4),   # top arc
                (angles >= -np.pi/4)   & (angles <  np.pi/4),   # right arc
                (angles >= np.pi/4)    & (angles <  3*np.pi/4), # bottom arc
                (angles >= 3*np.pi/4) | (angles < -3*np.pi/4),  # left arc
            ]
            mask = sector_masks[sector]
            if mask.any():
                dirs  = snake[mask] - np.array([cy, cx])
                norms = np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-8)
                delta = STEP * dirs / norms
                if expand:
                    snake[mask] += delta
                else:
                    snake[mask] -= delta
        # else: local action in Phase 1 → no-op (snake unchanged)

        snake[:, 0] = np.clip(snake[:, 0], 0, h - 1)
        snake[:, 1] = np.clip(snake[:, 1], 0, w - 1)

        self._snake      = snake
        self._step_count += 1

        truncated = self._step_count >= MAX_STEPS

        # Dense reward: Δ(IoU) − curvature penalty
        if self._gt_mask is not None:
            m = evaluate_snake(self._snake, self._gt_mask,
                               self._gt_contour, self._image.shape)
            iou = float(m["iou"])
        else:
            iou = 0.0
            m   = {}

        reward = iou - self._prev_iou   # pure Δ(IoU), no per-step drag
        self._prev_iou = iou

        # Terminal reward — only fires on timeout, not on done action
        # (done carries no bonus so the agent uses it purely to lock in a good state)
        if truncated:
            if iou > IOU_BONUS_THRESHOLD:
                reward += 1.0
            reward -= CURV_LAMBDA * float(_contour_curvature(self._snake))

        return self._observe(), float(reward), done, truncated, {}

    def _observe(self):
        mag, sdir, cdir = _gradient_features(self._image_smooth, self._snake, N_PROBES)
        stats = _contour_stats(self._snake, self._image.shape)
        phase = np.array([1.0 if self._step_count >= PHASE_SWITCH else 0.0],
                         dtype=np.float32)
        return np.concatenate([
            mag.astype(np.float32),
            sdir.astype(np.float32),
            cdir.astype(np.float32),
            stats,
            phase,
        ])

    def get_snake(self):
        return self._snake.copy()

    def get_metrics(self):
        if self._gt_mask is None:
            return {}
        return evaluate_snake(self._snake, self._gt_mask,
                              self._gt_contour, self._image.shape)

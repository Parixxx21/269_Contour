"""
GVF Snake — Method 5: Gradient Vector Flow active contour.

GVF (Xu & Prince, 1998) solves the short-range force problem by diffusing
the gradient field across the entire image via a PDE:

    ∂u/∂t = μ ∇²u − (u − u₀)|∇f|²
    ∂v/∂t = μ ∇²v − (v − v₀)|∇f|²

where f = |∇(Gσ*I)|² is the edge map, (u₀,v₀) = ∇f is the initial
force field, and μ controls the smoothness of the diffused field.

Result: a vector field that points toward edges everywhere in the image,
not just within ~3σ of an edge — so the snake converges from any
initialization without needing a large sigma or careful placement.

Snake optimization uses the same implicit time-stepping as AdaptiveSnake
but with uniform bending stiffness (no spatial adaptation).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, map_coordinates


class GVFSnake:
    """
    Active contour driven by Gradient Vector Flow external forces.

    Parameters
    ----------
    alpha       : elasticity weight
    beta        : uniform bending stiffness
    gamma       : implicit time-step (step ≈ F/gamma)
    sigma       : Gaussian smoothing for initial edge map
    mu          : GVF smoothness — larger → forces spread further but weaker
    gvf_iter    : diffusion iterations (more → longer range, slower)
    gvf_dt      : diffusion time step (must satisfy dt ≤ 1/(4μ) for stability)
    n_iter      : snake optimization iterations
    reparam_every: redistribute contour points every N iters (0 = off)
    """

    def __init__(
        self,
        alpha=0.015,
        beta=0.1,
        gamma=5.0,
        sigma=2.0,
        mu=0.2,
        gvf_iter=1000,
        gvf_dt=0.5,     # stable for mu≤0.25 (dt ≤ 1/4μ); gives spread ≈√(2μ·iter·dt) ≈20px
        n_iter=2000,
        reparam_every=50,
        wline=0.0,
        wedge=1.0,
    ):
        self.alpha        = alpha
        self.beta         = beta
        self.gamma        = gamma
        self.sigma        = sigma
        self.mu           = mu
        self.gvf_iter     = gvf_iter
        self.gvf_dt       = gvf_dt
        self.n_iter       = n_iter
        self.reparam_every = reparam_every
        self.wline        = wline
        self.wedge        = wedge

    # ── GVF computation ───────────────────────────────────────────────────────

    def _compute_gvf(self, image):
        """
        Compute GVF field (fy, fx) from the image.

        Steps
        -----
        1. Smooth image and compute edge map f = |∇I|²
        2. Initialise (u,v) = ∇f  (classical snake force field)
        3. Iterate the GVF PDE to diffuse forces across flat regions
        """
        img = gaussian_filter(image.astype(float), sigma=max(self.sigma, 0.1))

        # Edge map
        gy, gx = np.gradient(img)
        f  = self.wedge * (gx ** 2 + gy ** 2) + self.wline * img
        fy0, fx0 = np.gradient(f)   # initial force = ∇f

        u, v = fx0.copy(), fy0.copy()   # u→col force, v→row force

        dt, mu = self.gvf_dt, self.mu
        # Stability check: dt ≤ 1/(4μ)
        dt = min(dt, 1.0 / (4.0 * mu + 1e-8))

        for _ in range(self.gvf_iter):
            lap_u = laplace(u)
            lap_v = laplace(v)
            u = u + dt * (mu * lap_u - (u - fx0) * f)
            v = v + dt * (mu * lap_v - (v - fy0) * f)

        # Return as (fy, fx) matching (row, col) convention used by snakes
        return v, u

    # ── Stiffness matrix ──────────────────────────────────────────────────────

    def _build_matrix(self, n):
        """Uniform-beta pentadiagonal circulant stiffness matrix."""
        a, b = self.alpha, self.beta
        row = np.zeros(n)
        row[0]  =  2 * a + 6 * b
        row[1]  = -(a + 4 * b)
        row[-1] = -(a + 4 * b)
        row[2]  =  b
        row[-2] =  b
        from scipy.linalg import circulant
        return circulant(row)

    # ── Force interpolation ───────────────────────────────────────────────────

    @staticmethod
    def _force_at_snake(fy, fx, snake, h, w):
        y = np.clip(snake[:, 0], 0, h - 1)
        x = np.clip(snake[:, 1], 0, w - 1)
        return (
            map_coordinates(fy, [y, x], order=1, mode="nearest"),
            map_coordinates(fx, [y, x], order=1, mode="nearest"),
        )

    # ── Reparameterization ────────────────────────────────────────────────────

    @staticmethod
    def _reparameterize(snake):
        """Uniform arc-length redistribution (closed curve)."""
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

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, image, init_snake):
        """
        Parameters
        ----------
        image      : (H, W) float ndarray in [0, 1]
        init_snake : (N, 2) ndarray of (row, col) initial contour

        Returns
        -------
        snake   : (N, 2) final contour
        history : list of (N, 2) snapshots every 100 iterations
        """
        snake = init_snake.copy().astype(float)
        n     = len(snake)
        h, w  = image.shape

        # Compute GVF field (the expensive step — done once)
        fy, fx = self._compute_gvf(image)

        fscale = max(np.abs(fx).max(), np.abs(fy).max(), 1e-8)
        fx = fx / fscale
        fy = fy / fscale

        # Precompute inverse (uniform beta → matrix is fixed)
        A   = self._build_matrix(n)
        inv = np.linalg.inv(A + self.gamma * np.eye(n))

        history = [snake.copy()]

        for it in range(self.n_iter):
            fyn, fxn = self._force_at_snake(fy, fx, snake, h, w)

            yn = inv @ (self.gamma * snake[:, 0] + fyn)
            xn = inv @ (self.gamma * snake[:, 1] + fxn)

            snake[:, 0] = np.clip(yn, 0, h - 1)
            snake[:, 1] = np.clip(xn, 0, w - 1)

            if self.reparam_every > 0 and (it + 1) % self.reparam_every == 0:
                snake = self._reparameterize(snake)

            if (it + 1) % 100 == 0:
                history.append(snake.copy())

        return snake, history

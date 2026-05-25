import numpy as np
from scipy.linalg import circulant
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage.filters import gaussian

from .energy import compute_gradient_magnitude, compute_external_forces


class AdaptiveSnake:
    """
    Snake with spatially adaptive energy weighting.

    The bending stiffness beta_i at each contour point is modulated by the
    local gradient magnitude:

        beta_i = beta_min + (beta_max - beta_min) * exp(-k * |∇I(v_i)|)

    Near strong edges  → small beta  (contour deforms freely to track details)
    In flat/noisy areas → large beta  (strong regularization prevents drift)

    The elastic weight alpha and time-step gamma remain fixed globally.
    Update rule (implicit time-stepping):

        (A(beta) + gamma * I) v^{t+1} = gamma * v^t + F_ext(v^t)
    """

    def __init__(
        self,
        alpha=0.015,
        beta_min=0.005,
        beta_max=0.05,
        k=10.0,           # sensitivity to gradient magnitude
        gamma=5.0,        # implicit time-step (global)
        sigma=6.0,        # smoothing for external energy
        n_iter=800,
        update_every=10,  # recompute adaptive matrix every N iterations
        reparam_every=50, # arc-length redistribution every N iterations (0 = off)
        wline=0.0,
        wedge=1.0,
    ):
        self.alpha = alpha
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.k = k
        self.gamma = gamma
        self.sigma = sigma
        self.n_iter = n_iter
        self.update_every = update_every
        self.reparam_every = reparam_every
        self.wline = wline
        self.wedge = wedge

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adaptive_beta(self, image, snake):
        """Per-point β: β_min + (β_max - β_min)*exp(-k*|∇I|)."""
        gmag = compute_gradient_magnitude(image, sigma=self.sigma)
        h, w = gmag.shape
        y = np.clip(snake[:, 0], 0, h - 1)
        x = np.clip(snake[:, 1], 0, w - 1)
        local_g = map_coordinates(gmag, [y, x], order=1, mode="nearest")
        return self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.k * local_g)

    def _build_matrix(self, n, beta_array):
        """
        Build the n×n position-dependent stiffness matrix.
        Row i uses local alpha and beta_i (local stencil approximation).
        """
        A = np.zeros((n, n))
        a = self.alpha
        for i in range(n):
            b = beta_array[i]
            A[i, i] = 2 * a + 6 * b
            A[i, (i + 1) % n] = -(a + 4 * b)
            A[i, (i - 1) % n] = -(a + 4 * b)
            A[i, (i + 2) % n] = b
            A[i, (i - 2) % n] = b
        return A

    def _force_at_snake(self, fy, fx, snake, h, w):
        """Bilinearly interpolate external forces at snake points."""
        y = np.clip(snake[:, 0], 0, h - 1)
        x = np.clip(snake[:, 1], 0, w - 1)
        coords = [y, x]
        return (
            map_coordinates(fy, coords, order=1, mode="nearest"),
            map_coordinates(fx, coords, order=1, mode="nearest"),
        )

    @staticmethod
    def _reparameterize(snake):
        """Redistribute snake points uniformly by arc length (closed curve)."""
        # Arc is measured from snake[0]; the closing segment snake[-1]→snake[0]
        # is intentionally excluded from redistribution to avoid placing new
        # points into potentially noisy inter-level regions.
        diffs = np.diff(snake, axis=0, prepend=snake[[-1]])
        arc = np.cumsum(np.linalg.norm(diffs, axis=1))
        arc -= arc[0]   # arc[0]=0, arc[-1] = open-curve length (excl. closing segment)
        total = arc[-1]
        if total < 1e-8:
            return snake
        uniform = np.linspace(0, total, len(snake), endpoint=False)
        arc_ext = np.append(arc, total + arc[1])  # small extension for edge interpolation
        snake_ext = np.vstack([snake, snake[0]])
        new_y = np.interp(uniform, arc_ext, snake_ext[:, 0])
        new_x = np.interp(uniform, arc_ext, snake_ext[:, 1])
        return np.column_stack([new_y, new_x])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, image, init_snake):
        """
        Parameters
        ----------
        image      : (H, W) float ndarray in [0, 1]
        init_snake : (N, 2) ndarray of (row, col) initial points

        Returns
        -------
        snake   : (N, 2) final contour
        history : list of (N, 2) snapshots every update_every iterations
        """
        snake = init_snake.copy().astype(float)
        n = len(snake)
        h, w = image.shape

        # External force field (fixed — computed once on smoothed image)
        fy, fx = compute_external_forces(image, sigma=self.sigma,
                                         wline=self.wline, wedge=self.wedge)

        # Normalize forces to avoid excessively large steps
        fscale = max(np.abs(fx).max(), np.abs(fy).max(), 1e-8)
        fx = fx / fscale
        fy = fy / fscale

        history = [snake.copy()]
        inv = None

        for it in range(self.n_iter):
            if it % self.update_every == 0:
                beta_arr = self._adaptive_beta(image, snake)
                A = self._build_matrix(n, beta_arr)
                inv = np.linalg.inv(A + self.gamma * np.eye(n))

            fyn, fxn = self._force_at_snake(fy, fx, snake, h, w)

            yn = inv @ (self.gamma * snake[:, 0] + fyn)
            xn = inv @ (self.gamma * snake[:, 1] + fxn)

            snake[:, 0] = np.clip(yn, 0, h - 1)
            snake[:, 1] = np.clip(xn, 0, w - 1)

            if self.reparam_every > 0 and (it + 1) % self.reparam_every == 0:
                snake = self._reparameterize(snake)

            if (it + 1) % self.update_every == 0:
                history.append(snake.copy())

        return snake, history

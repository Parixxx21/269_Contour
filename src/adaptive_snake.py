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

        beta_i = beta_max * exp(-k * |∇I(v_i)|)

    Near strong edges  → small beta  (contour deforms freely to track details)
    In flat/noisy areas → large beta (strong regularization prevents drift)

    The elastic weight alpha remains fixed.  The update rule uses implicit
    time-stepping:

        (A(beta) + gamma * I) v^{t+1} = gamma * v^t + F_ext(v^t)

    where A(beta) is the position-dependent pentadiagonal stiffness matrix.
    """

    def __init__(
        self,
        alpha=0.015,
        beta_min=0.005,
        beta_max=0.3,
        k=5.0,           # sensitivity to gradient magnitude
        gamma=100.0,     # implicit time-step (larger = more stable, smaller steps)
        sigma=2.0,       # smoothing for external energy
        n_iter=500,
        update_every=20, # recompute adaptive matrix every N iterations
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
        self.wline = wline
        self.wedge = wedge

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adaptive_beta(self, image, snake):
        """Return per-point beta values based on local gradient magnitude."""
        gmag = compute_gradient_magnitude(image, sigma=self.sigma)
        h, w = gmag.shape
        y = np.clip(snake[:, 0], 0, h - 1)
        x = np.clip(snake[:, 1], 0, w - 1)
        local_g = map_coordinates(gmag, [y, x], order=1, mode="nearest")
        beta = self.beta_max * np.exp(-self.k * local_g)
        return np.clip(beta, self.beta_min, self.beta_max)

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

            if (it + 1) % self.update_every == 0:
                history.append(snake.copy())

        return snake, history

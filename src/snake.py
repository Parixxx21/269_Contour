import numpy as np
from scipy.ndimage import map_coordinates

from .energy import compute_external_forces


class ClassicalSnake:
    """
    Classical parametric active contour (snake) with fixed regularization.

    Minimizes:
        E = alpha * E_elastic + beta * E_smooth + gamma * E_image

    where internal energy terms enforce contour regularity and the external
    energy term attracts the contour toward image edges/lines.

    The implementation uses the same implicit time-stepping structure as the
    adaptive model, but keeps beta fixed for every contour point:

        (A(alpha, beta) + gamma * I) v^{t+1} = gamma * v^t + F_ext(v^t)

    Reference: Kass, Witkin & Terzopoulos, "Snakes: Active Contour Models" (1988).
    """

    def __init__(
        self,
        alpha=0.02,    # w1: elasticity weight (resistance to stretching)
        beta=0.05,     # w2: bending stiffness weight (resistance to curvature), fixed globally
        gamma=2.5,     # D/Δt where D = γI is the scalar damping matrix (Kass et al.)
        sigma=8.0,     # Gaussian smoothing scale for external potential P(x,y)
        n_iter=1600,
        update_every=20,
        reparam_every=50,
        wline=0.0,     # weight on line (intensity) energy
        wedge=1.0,     # weight on edge (gradient) energy
        boundary_condition="periodic",
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.n_iter = n_iter
        self.update_every = update_every
        self.reparam_every = reparam_every
        self.wline = wline
        self.wedge = wedge
        self.boundary_condition = boundary_condition

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_matrix(self, n):
        """Build K: the n×n stiffness matrix from w1 (alpha) and w2 (beta)."""
        if self.boundary_condition != "periodic":
            raise ValueError("ClassicalSnake currently supports only periodic contours")

        A = np.zeros((n, n))  # K in Kass et al.
        a = self.alpha
        b = self.beta
        for i in range(n):
            A[i, i] = 2 * a + 6 * b
            A[i, (i + 1) % n] = -(a + 4 * b)
            A[i, (i - 1) % n] = -(a + 4 * b)
            A[i, (i + 2) % n] = b
            A[i, (i - 2) % n] = b
        return A

    @staticmethod
    def _force_at_snake(fy, fx, snake, h, w):
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
        closed = np.vstack([snake, snake[0]])
        seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = arc[-1]
        if total < 1e-8:
            return snake

        uniform = np.linspace(0, total, len(snake), endpoint=False)
        new_y = np.interp(uniform, arc, closed[:, 0])
        new_x = np.interp(uniform, arc, closed[:, 1])
        return np.column_stack([new_y, new_x])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, image, init_snake):
        """
        Fit the snake to the image.

        Parameters
        ----------
        image : (H, W) ndarray, grayscale float in [0, 1]
        init_snake : (N, 2) ndarray of (row, col) initial contour points

        Returns
        -------
        snake : (N, 2) ndarray, optimized contour in (row, col)
        history : list of (N, 2) arrays sampled every update_every iterations
        """
        snake = init_snake.copy().astype(float)
        n = len(snake)
        h, w = image.shape

        fy, fx = compute_external_forces(
            image, sigma=self.sigma, wline=self.wline, wedge=self.wedge
        )
        fscale = max(np.abs(fx).max(), np.abs(fy).max(), 1e-8)
        fx = fx / fscale
        fy = fy / fscale

        A = self._build_matrix(n)                   # K: stiffness matrix
        inv = np.linalg.inv(A + self.gamma * np.eye(n))  # (K + D/Δt)^{-1}
        history = [snake.copy()]

        for it in range(self.n_iter):
            fyn, fxn = self._force_at_snake(fy, fx, snake, h, w)

            yn = inv @ (self.gamma * snake[:, 0] + fyn)  # solve (K + D/Δt)c^{t+1} = D/Δt·c^t + f
            xn = inv @ (self.gamma * snake[:, 1] + fxn)

            snake[:, 0] = np.clip(yn, 0, h - 1)
            snake[:, 1] = np.clip(xn, 0, w - 1)

            if self.reparam_every > 0 and (it + 1) % self.reparam_every == 0:
                snake = self._reparameterize(snake)

            if (it + 1) % self.update_every == 0:
                history.append(snake.copy())

        return snake, history
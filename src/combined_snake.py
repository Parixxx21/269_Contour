import numpy as np
from skimage.filters import gaussian

from .adaptive_snake import AdaptiveSnake
from .energy import compute_external_forces


class CombinedSnake(AdaptiveSnake):
    """
    Combined coarse-to-fine + spatially adaptive snake.

    At each pyramid level the contour is refined using the adaptive implicit
    time-stepping solver (position-dependent beta).  Moving from coarse to
    fine levels gives a strong warm-start initialization while the adaptive
    weights handle heterogeneous edge strength within each level.

    Inherits all adaptive parameters from AdaptiveSnake.
    Additional parameters control the multi-scale pyramid.
    """

    def __init__(
        self,
        sigma_coarse=8.0,
        n_levels=3,
        n_iter_per_level=None,
        **adaptive_kwargs,
    ):
        super().__init__(**adaptive_kwargs)
        self.sigma_coarse = sigma_coarse
        self.n_levels = n_levels
        # Distribute total iterations across levels
        total = adaptive_kwargs.get("n_iter", 500)
        self.n_iter_per_level = n_iter_per_level or max(1, total // n_levels)

    def _build_pyramid(self, image):
        """Coarsest-first list of smoothed images."""
        sigmas = np.linspace(self.sigma_coarse, 0.5, self.n_levels)
        return [gaussian(image.astype(float), sigma=s) for s in sigmas]

    def fit(self, image, init_snake):
        """
        Parameters
        ----------
        image      : (H, W) float ndarray in [0, 1]
        init_snake : (N, 2) ndarray of (row, col) initial contour

        Returns
        -------
        snake   : (N, 2) final contour
        history : list of (N, 2) snapshots across all pyramid levels
        """
        pyramid = self._build_pyramid(image)
        snake = init_snake.copy().astype(float)
        n = len(snake)
        h, w = image.shape
        history = [snake.copy()]

        orig_n_iter = self.n_iter
        self.n_iter = self.n_iter_per_level

        for level_img in pyramid:
            # External forces computed on level image but adaptive weights use original
            fy, fx = compute_external_forces(
                level_img, sigma=self.sigma, wline=self.wline, wedge=self.wedge
            )
            fscale = max(np.abs(fx).max(), np.abs(fy).max(), 1e-8)
            fx = fx / fscale
            fy = fy / fscale

            inv = None
            for it in range(self.n_iter_per_level):
                if it % self.update_every == 0:
                    # Adaptive weights evaluated on the original (non-blurred) image
                    beta_arr = self._adaptive_beta(image, snake)
                    A = self._build_matrix(n, beta_arr)
                    inv = np.linalg.inv(A + self.gamma * np.eye(n))

                fyn, fxn = self._force_at_snake(fy, fx, snake, h, w)
                yn = inv @ (self.gamma * snake[:, 0] + fyn)
                xn = inv @ (self.gamma * snake[:, 1] + fxn)
                snake[:, 0] = np.clip(yn, 0, h - 1)
                snake[:, 1] = np.clip(xn, 0, w - 1)

            history.append(snake.copy())

        self.n_iter = orig_n_iter
        return snake, history

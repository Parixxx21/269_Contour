import numpy as np
from skimage.filters import gaussian

from .energy import compute_external_forces
from .snake import ClassicalSnake


class MultiscaleSnake(ClassicalSnake):
    """
    Coarse-to-fine snake using a Gaussian image pyramid.

    Strategy
    --------
    1. Build a pyramid of n_levels images, each progressively less smoothed.
       Level 0 is the most blurred (convex energy landscape, easy to optimize).
       Level n-1 is the finest (original-scale detail).
    2. Run the fixed-beta implicit snake at each level, propagating the contour
       from coarser to finer levels as a warm start.

    This significantly reduces the chance of the contour settling in a local
    minimum caused by noise or weak edges.

    This class uses the same fixed-beta implicit solver as ClassicalSnake at
    each pyramid level.
    """

    def __init__(
        self,
        alpha=0.015,
        beta=0.05,
        gamma=3.0,
        sigma_coarse=8.0,
        sigma_fine=3.0,
        n_levels=3,
        n_iter_per_level=None,
        n_iter=2400,
        update_every=10,
        reparam_every=50,
        wline=0.0,
        wedge=1.0,
        boundary_condition="periodic",
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            sigma=sigma_fine,
            n_iter=n_iter,
            update_every=update_every,
            reparam_every=reparam_every,
            wline=wline,
            wedge=wedge,
            boundary_condition=boundary_condition,
        )
        self.sigma_coarse = sigma_coarse
        self.sigma_fine = sigma_fine
        self.n_levels = n_levels
        # Weight iterations toward coarse levels so elastic propagation has time
        # to pull far-away points into the force field before refining.
        if n_iter_per_level is not None:
            self.iters_per_level = [n_iter_per_level] * n_levels
        else:
            base = max(1, n_iter // n_levels)
            self.iters_per_level = [base * (n_levels - i) // n_levels
                                    for i in range(n_levels)]
            self.iters_per_level[-1] = max(base, n_iter - sum(self.iters_per_level[:-1]))

    def _build_pyramid(self, image):
        """Return list of smoothed images, coarsest first."""
        sigmas = np.linspace(self.sigma_coarse, self.sigma_fine, self.n_levels)
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
        history : list of (N, 2) snapshots, one per pyramid level checkpoint
        """
        pyramid = self._build_pyramid(image)
        snake = init_snake.copy().astype(float)
        n = len(snake)
        h, w = image.shape
        history = [snake.copy()]
        A = self._build_matrix(n)
        inv = np.linalg.inv(A + self.gamma * np.eye(n))

        for level_img, n_iter in zip(pyramid, self.iters_per_level):
            # Pyramid image is already smoothed, so use minimal extra smoothing
            # when deriving the force field for this level.
            fy, fx = compute_external_forces(
                level_img, sigma=0.5, wline=self.wline, wedge=self.wedge
            )
            fscale = max(np.abs(fx).max(), np.abs(fy).max(), 1e-8)
            fx = fx / fscale
            fy = fy / fscale

            for it in range(n_iter):
                fyn, fxn = self._force_at_snake(fy, fx, snake, h, w)
                yn = inv @ (self.gamma * snake[:, 0] + fyn)
                xn = inv @ (self.gamma * snake[:, 1] + fxn)

                snake[:, 0] = np.clip(yn, 0, h - 1)
                snake[:, 1] = np.clip(xn, 0, w - 1)

                if self.reparam_every > 0 and (it + 1) % self.reparam_every == 0:
                    snake = self._reparameterize(snake)

            history.append(snake.copy())

        return snake, history

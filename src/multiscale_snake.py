import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian


class MultiscaleSnake:
    """
    Coarse-to-fine snake using a Gaussian image pyramid.

    Strategy
    --------
    1. Build a pyramid of n_levels images, each progressively less smoothed.
       Level 0 is the most blurred (convex energy landscape, easy to optimize).
       Level n-1 is the finest (original-scale detail).
    2. Run skimage's active_contour at each level, propagating the contour
       from coarser to finer levels as a warm start.

    This significantly reduces the chance of the contour settling in a local
    minimum caused by noise or weak edges.
    """

    def __init__(
        self,
        alpha=0.015,
        beta=0.1,
        gamma=0.001,
        sigma_coarse=4.0,
        sigma_fine=1.5,
        n_levels=3,
        n_iter_per_level=None,
        n_iter=2500,
        wline=0.0,
        wedge=1.0,
        boundary_condition="periodic",
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma_coarse = sigma_coarse
        self.sigma_fine = sigma_fine
        self.n_levels = n_levels
        self.n_iter = n_iter
        # Weight iterations toward coarse levels so elastic propagation has time
        # to pull far-away points into the force field before refining.
        if n_iter_per_level is not None:
            self.iters_per_level = [n_iter_per_level] * n_levels
        else:
            base = max(1, n_iter // n_levels)
            self.iters_per_level = [base * (n_levels - i) // n_levels
                                    for i in range(n_levels)]
            self.iters_per_level[-1] = max(base, n_iter - sum(self.iters_per_level[:-1]))
        self.wline = wline
        self.wedge = wedge
        self.boundary_condition = boundary_condition

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
        history = [snake.copy()]

        for level_img, n_iter in zip(pyramid, self.iters_per_level):
            snake = active_contour(
                level_img,
                snake,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                w_line=self.wline,
                w_edge=self.wedge,
                max_num_iter=n_iter,
                boundary_condition=self.boundary_condition,
            )
            history.append(snake.copy())

        return snake, history

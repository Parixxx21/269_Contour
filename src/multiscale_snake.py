import numpy as np
from skimage.transform import rescale

from .snake import ClassicalSnake


class MultiscaleSnake(ClassicalSnake):
    """
    Coarse-to-fine snake using a true spatial image pyramid.

    This version keeps the implicit ClassicalSnake solver and improves
    robustness by optimizing on progressively finer image resolutions.
    """

    def __init__(
        self,
        n_levels=3,
        n_iter_per_level=None,
        sigma_coarse=None,
        sigma_fine=None,
        **classical_kwargs,
    ):
        sigma = classical_kwargs.pop("sigma", None)
        if sigma is None:
            sigma = sigma_fine if sigma_fine is not None else 8.0
        classical_kwargs["sigma"] = sigma

        super().__init__(**classical_kwargs)
        self.n_levels = n_levels
        self.sigma_coarse = sigma_coarse if sigma_coarse is not None else self.sigma
        self.sigma_fine = sigma_fine if sigma_fine is not None else self.sigma

        total_iter = classical_kwargs.get("n_iter", 2500)
        self.n_iter_per_level = n_iter_per_level or max(1, total_iter // n_levels)

    def _build_spatial_pyramid(self, image):
        """
        Construct a multi-scale spatial image pyramid via anti-aliased downsampling.
        Returns a list of scaled images ordered from COARSE to FINE.
        """
        pyramid = [image]
        current_img = image

        for _ in range(self.n_levels - 1):
            current_img = rescale(current_img, 0.5, anti_aliasing=True)
            pyramid.append(current_img)

        return pyramid[::-1]

    def fit(self, image, init_snake):
        """
        Parameters
        ----------
        image      : (H, W) float ndarray in [0, 1]
        init_snake : (N, 2) ndarray of (row, col) initial contour

        Returns
        -------
        snake   : (N, 2) final optimized contour on the original scale
        history : list of (N, 2) snapshots mapped back to original coordinates
        """
        pyramid = self._build_spatial_pyramid(image)
        sigma_levels = np.linspace(self.sigma_coarse, self.sigma_fine, self.n_levels)

        scale_factor = 0.5 ** (self.n_levels - 1)
        snake = init_snake.copy().astype(float) * scale_factor
        history = [init_snake.copy()]

        original_n_iter = self.n_iter
        original_sigma = self.sigma

        for level_idx, (level_img, level_sigma) in enumerate(zip(pyramid, sigma_levels)):
            self.n_iter = self.n_iter_per_level
            self.sigma = float(level_sigma)

            snake, _ = super().fit(level_img, snake)

            current_scale_up = 2 ** (self.n_levels - 1 - level_idx)
            history.append(snake.copy() * current_scale_up)

            if level_idx < self.n_levels - 1:
                snake = snake * 2.0

        self.n_iter = original_n_iter
        self.sigma = original_sigma
        return snake, history

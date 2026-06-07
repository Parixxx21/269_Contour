import numpy as np
from skimage.transform import rescale

# Import the baseline ClassicalSnake to maintain programmatic consistency
from .snake import ClassicalSnake


class MultiscaleSnake(ClassicalSnake):
    """
    Coarse-to-fine snake using a true spatial image pyramid.

    This class constructs a downsampled spatial pyramid of the input image.
    The snake starts optimization on a highly downsampled (coarse) resolution 
    to bypass local minima and noise, then systematically upscales its coordinates 
    to act as a warm-start initialization for finer resolution levels.
    """

    def __init__(
        self,
        n_levels=3,
        n_iter_per_level=None,
        **classical_kwargs,
    ):
        # Explicitly ensure stable execution by passing uniform baseline parameters
        super().__init__(**classical_kwargs)
        self.n_levels = n_levels
        
        # Distribute the total computational iteration budget evenly across all levels
        total_iter = classical_kwargs.get("n_iter", 2500)
        self.n_iter_per_level = n_iter_per_level or max(1, total_iter // n_levels)

    def _build_spatial_pyramid(self, image):
        """
        Construct a multi-scale spatial image pyramid via anti-aliased downsampling.
        Returns a list of scaled images ordered from COARSE (low-res) to FINE (original-res).
        """
        pyramid = [image]
        current_img = image
        
        # Progressively downsample the image scale by a factor of 0.5
        for _ in range(self.n_levels - 1):
            # rescale with anti_aliasing=True automatically applies an appropriate
            # Gaussian blur before downsampling to prevent high-frequency artifacts.
            current_img = rescale(current_img, 0.5, anti_aliasing=True)
            pyramid.append(current_img)
            
        # Reverse the list layout from [Fine, Medium, Coarse] to [Coarse, Medium, Fine]
        return pyramid[::-1]

    def fit(self, image, init_snake):
        """
        Parameters
        ----------
        image      : (H, W) float ndarray in [0, 1]
        init_snake : (N, 2) ndarray of (row, col) initial contour setup

        Returns
        -------
        snake   : (N, 2) final optimized contour on the original scale
        history : list of (N, 2) snapshots mapped back to the original image coordinates
        """
        pyramid = self._build_spatial_pyramid(image)
        
        # Scale down the initial coordinate layout to align with the coarsest pyramid layer
        scale_factor = 0.5 ** (self.n_levels - 1)
        snake = init_snake.copy().astype(float) * scale_factor
        
        # Initialize history array with the downscaled tracking starting point
        history = [init_snake.copy()]

        # Progress through the spatial resolution layers
        for level_idx, level_img in enumerate(pyramid):
            
            # Temporary adjustments to match the fit signature of the baseline solver.
            # We preserve total iteration allocation per level during this loop phase.
            original_n_iter = self.n_iter
            self.n_iter = self.n_iter_per_level
            
            # Execute the baseline unified implicit solver on the current resolution level image
            # ClassicalSnake internally handles compute_external_forces and scaling.
            snake, _ = super().fit(level_img, snake)
            
            # Restore class property integrity
            self.n_iter = original_n_iter

            # Project current checkpoint coordinates back onto original matrix resolution for tracking
            current_scale_up = 2 ** (self.n_levels - 1 - level_idx)
            history.append(snake.copy() * current_scale_up)

            # Unless tracking has reached native resolution limits, double the contour scale factor
            if level_idx < self.n_levels - 1:
                snake = snake * 2.0

        return snake, history
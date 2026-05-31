import numpy as np
from skimage.filters import gaussian
from skimage.transform import rescale

from .adaptive_snake import AdaptiveSnake
from .energy import compute_external_forces


class CombinedSnake(AdaptiveSnake):
    """
    Combined coarse-to-fine (multi-scale) + spatially adaptive snake.

    1. Coarse-to-Fine: Constructs a spatial image pyramid. The snake is initialized
       on a low-resolution image to bypass local minima, then upscaled to higher resolutions.
    2. Adaptive: At each level, the bending stiffness (beta) adapts to the local 
       gradient of that specific resolution level.
    """

    def __init__(
        self,
        n_levels=3,
        n_iter_per_level=None,
        **adaptive_kwargs,
    ):
        super().__init__(**adaptive_kwargs)
        self.n_levels = n_levels
        
        self.n_iter_per_level = n_iter_per_level or max(1, self.n_iter // n_levels)

    def _build_spatial_pyramid(self, image):
        """
        Step 1: Construct a multi-scale spatial pyramid.
        Returns a list of images ordered from COARSE (low-res) to FINE (original-res).
        """
        pyramid = [image]
        current_img = image
        
        # Iteratively downsample the image by a factor of 0.5
        for _ in range(self.n_levels - 1):
            # rescale with anti_aliasing=True automatically applies a Gaussian blur
            # before shrinking, which prevents sampling artifacts.
            current_img = rescale(current_img, 0.5, anti_aliasing=True)
            pyramid.append(current_img)
            
        # The list is currently [Fine, Medium, Coarse]. 
        # We reverse it to [Coarse, Medium, Fine] for the optimization loop.
        return pyramid[::-1]

    def fit(self, image, init_snake):
        """
        Step 2: The Coarse-to-Fine Optimization Loop
        """
        pyramid = self._build_spatial_pyramid(image)
        
        # Scale the initial contour to match the coarsest (smallest) image level
        # If n_levels = 3, scale_factor = 0.5^2 = 0.25
        scale_factor = 0.5 ** (self.n_levels - 1)
        snake = init_snake.copy().astype(float) * scale_factor
        
        history = []

        # Iterate from Coarse (low-res) to Fine (high-res)
        for level_idx, level_img in enumerate(pyramid):
            h, w = level_img.shape
            
            # Compute forces for the CURRENT resolution level
            # We use a small sigma because the image is already smoothed from downsampling
            fy, fx = compute_external_forces(
                level_img, sigma=self.sigma, wline=self.wline, wedge=self.wedge
            )
            
            # Normalize forces
            fscale = max(np.abs(fx).max(), np.abs(fy).max(), 1e-8)
            fx = fx / fscale
            fy = fy / fscale

            inv = None
            
            # Optimization loop for the current level
            for it in range(self.n_iter_per_level):
                # IMPORTANT FIX: Pass `level_img` instead of `image` so beta calculates correctly
                if it % self.update_every == 0:
                    beta_arr = self._adaptive_beta(level_img, snake)
                    A = self._build_matrix(len(snake), beta_arr)
                    inv = np.linalg.inv(A + self.gamma * np.eye(len(snake)))

                fyn, fxn = self._force_at_snake(fy, fx, snake, h, w)
                
                yn = inv @ (self.gamma * snake[:, 0] + fyn)
                xn = inv @ (self.gamma * snake[:, 1] + fxn)
                
                snake[:, 0] = np.clip(yn, 0, h - 1)
                snake[:, 1] = np.clip(xn, 0, w - 1)

                if self.reparam_every > 0 and (it + 1) % self.reparam_every == 0:
                    snake = self._reparameterize(snake)

            # Save the final contour of this level to history
            # (We multiply by the inverse scale so history is easily visualizable on the original image)
            current_scale_up = 2 ** (self.n_levels - 1 - level_idx)
            history.append(snake.copy() * current_scale_up)

            # If this is not the last level (original image), upscale the snake 
            # coordinates by x2 to act as the initialization for the next, larger level.
            if level_idx < self.n_levels - 1:
                snake = snake * 2.0

        return snake, history
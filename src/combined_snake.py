import numpy as np
from skimage.filters import gaussian
from skimage.transform import rescale

from .adaptive_snake import AdaptiveSnake
from .energy import compute_external_forces


class CombinedSnake(AdaptiveSnake):
    """
    Optimized coarse-to-fine (multi-scale) + spatially adaptive snake
    tailored for faint, real-world low-contrast biomedical images (e.g., HeLa cells).
    """

    def __init__(
        self,
        n_levels=3,
        n_iter_per_level=None,
        **adaptive_kwargs,
    ):
        super().__init__(**adaptive_kwargs)
        self.n_levels = n_levels
        # Base budget allocation per level
        self.n_iter_per_level = n_iter_per_level or max(1, self.n_iter // n_levels)

    def _build_spatial_pyramid(self, image):
        """Construct a multi-scale spatial pyramid ordered from COARSE to FINE."""
        pyramid = [image]
        current_img = image
        
        for _ in range(self.n_levels - 1):
            current_img = rescale(current_img, 0.5, anti_aliasing=True)
            pyramid.append(current_img)
            
        return pyramid[::-1]

    def fit(self, image, init_snake):
        """Optimized Coarse-to-Fine Loop with Adaptive Resolution Constraints."""
        pyramid = self._build_spatial_pyramid(image)
        
        scale_factor = 0.5 ** (self.n_levels - 1)
        snake = init_snake.copy().astype(float) * scale_factor
        
        history = []

        for level_idx, level_img in enumerate(pyramid):
            h, w = level_img.shape
            
            # Dynamic Level-Dependent Sigma Scaling
            # Coarse levels are already downsampled and blurred by rescale(). 
            # Re-applying a heavy global sigma on a tiny image completely annihilates small cell edges.
            # We dynamically shrink sigma at coarser resolutions to preserve faint gradients.
            if level_idx == self.n_levels - 1:
                current_sigma = self.sigma  # Use native fine-detail sigma at original scale
            else:
                current_sigma = max(0.5, self.sigma * 0.5)  # Constrain blur at coarse scales
            
            # Compute forces using the resolution-calibrated sigma
            fy, fx = compute_external_forces(
                level_img, sigma=current_sigma, wline=self.wline, wedge=self.wedge
            )
            
            # Normalize edge forces
            fscale = max(np.abs(fx).max(), np.abs(fy).max(), 1e-8)
            fx = fx / fscale
            fy = fy / fscale

            # Final Level External Force Booster
            # Real cell membranes have fuzzy gradients. Boosting the external edge force multiplier 
            # exclusively at the original native resolution layer forces the snake to snap tight.
            if level_idx == self.n_levels - 1:
                fx *= 1.5
                fy *= 1.5

            # Progressive Computational Budget Allocation
            # Coarse tracking only needs a quick rough sketch. The final fine level requires 
            # a massive operational budget to creep into sub-pixel microscopic cellular boundaries.
            if level_idx == self.n_levels - 1:
                level_iters = int(self.n_iter_per_level * 1.5)  # Boost iterations at native scale
            else:
                level_iters = int(self.n_iter_per_level * 0.7)  # Truncate iterations at coarse scales

            inv = None
            
            # Optimization loop for the current level
            for it in range(level_iters):
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

            # Save historical checkpoints scaled back to the reference frame
            current_scale_up = 2 ** (self.n_levels - 1 - level_idx)
            history.append(snake.copy() * current_scale_up)

            if level_idx < self.n_levels - 1:
                snake = snake * 2.0

        return snake, history
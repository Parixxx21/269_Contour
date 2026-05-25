"""
FullAdaptiveSnake: per-point α_i, β_i, γ_i all modulated by local gradient.

Extends AdaptiveSnake by also making γ spatially adaptive:

    β_i = β_min + (β_max - β_min) * exp(-k_β  * |∇I(v_i)|)
    γ_i = γ_min + (γ_max - γ_min) * exp(-k_γ  * |∇I(v_i)|)

Near strong edges  → small β_i, small γ_i  (deforms freely, follows force)
In flat/noisy areas → large β_i, large γ_i  (stiff, resists drift)

Update rule:
    (A(β) + diag(γ)) v^{t+1} = diag(γ) v^t + F_ext(v^t)
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

from .energy import compute_external_forces, compute_gradient_magnitude
from .adaptive_snake import AdaptiveSnake   # reuse helpers


class FullAdaptiveSnake(AdaptiveSnake):
    """
    AdaptiveSnake + per-point γ_i.

    All AdaptiveSnake parameters apply; adds:
        gamma_min  : γ at strong edges (aggressive, follows force)
        gamma_max  : γ in flat/noisy areas (conservative, resists drift)
        k_gamma    : sensitivity of γ to gradient magnitude
    """

    def __init__(
        self,
        alpha=0.015,
        beta_min=0.005,
        beta_max=0.3,
        k=5.0,
        gamma_min=2.0,
        gamma_max=10.0,
        k_gamma=5.0,
        sigma=8.0,
        n_iter=2500,
        update_every=20,
        reparam_every=50,
        wline=0.0,
        wedge=1.0,
    ):
        super().__init__(
            alpha=alpha,
            beta_min=beta_min,
            beta_max=beta_max,
            k=k,
            gamma=gamma_max,   # parent stores gamma; not used in our override
            sigma=sigma,
            n_iter=n_iter,
            update_every=update_every,
            reparam_every=reparam_every,
            wline=wline,
            wedge=wedge,
        )
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.k_gamma   = k_gamma

    def _adaptive_gamma(self, local_g):
        """Per-point γ: γ_min + (γ_max - γ_min)*exp(-k_γ*|∇I|)."""
        return self.gamma_min + (self.gamma_max - self.gamma_min) * np.exp(-self.k_gamma * local_g)

    def fit(self, image, init_snake):
        snake = init_snake.copy().astype(float)
        n = len(snake)
        h, w = image.shape

        fy, fx = compute_external_forces(image, sigma=self.sigma,
                                         wline=self.wline, wedge=self.wedge)
        fscale = max(np.abs(fx).max(), np.abs(fy).max(), 1e-8)
        fx /= fscale
        fy /= fscale

        history = [snake.copy()]
        inv = None
        gamma_arr = None

        for it in range(self.n_iter):
            if it % self.update_every == 0:
                gmag     = compute_gradient_magnitude(image, sigma=self.sigma)
                y_c = np.clip(snake[:, 0], 0, h - 1)
                x_c = np.clip(snake[:, 1], 0, w - 1)
                local_g  = map_coordinates(gmag, [y_c, x_c], order=1, mode="nearest")

                beta_arr  = self._adaptive_beta(image, snake)
                gamma_arr = self._adaptive_gamma(local_g)
                A   = self._build_matrix(n, beta_arr)
                inv = np.linalg.inv(A + np.diag(gamma_arr))

            fyn, fxn = self._force_at_snake(fy, fx, snake, h, w)

            snake[:, 0] = np.clip(inv @ (gamma_arr * snake[:, 0] + fyn), 0, h - 1)
            snake[:, 1] = np.clip(inv @ (gamma_arr * snake[:, 1] + fxn), 0, w - 1)

            if self.reparam_every > 0 and (it + 1) % self.reparam_every == 0:
                snake = self._reparameterize(snake)

            if (it + 1) % self.update_every == 0:
                history.append(snake.copy())

        return snake, history

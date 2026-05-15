import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian


class ClassicalSnake:
    """
    Classical parametric active contour (snake) using skimage's active_contour.

    Minimizes:
        E = alpha * E_elastic + beta * E_smooth + gamma * E_image

    where internal energy terms enforce contour regularity and the external
    energy term attracts the contour toward image edges/lines.

    Reference: Kass, Witkin & Terzopoulos, "Snakes: Active Contour Models" (1988).
    skimage docs: skimage.segmentation.active_contour
    """

    def __init__(
        self,
        alpha=0.015,   # elasticity (resistance to stretching)
        beta=0.1,      # bending stiffness (resistance to curvature)
        gamma=0.001,   # time step / gradient descent step size
        sigma=2.0,     # Gaussian smoothing for external energy
        n_iter=2500,
        wline=0.0,     # weight on line (intensity) energy
        wedge=1.0,     # weight on edge (gradient) energy
        boundary_condition="periodic",
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.n_iter = n_iter
        self.wline = wline
        self.wedge = wedge
        self.boundary_condition = boundary_condition

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
        history : list of (N, 2) arrays sampled every 250 iterations
        """
        img_smooth = gaussian(image.astype(float), sigma=self.sigma)

        history = []
        record_every = max(1, self.n_iter // 10)

        # skimage active_contour expects (N, 2) in (row, col) = (y, x)
        snake = active_contour(
            img_smooth,
            init_snake.copy(),
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            w_line=self.wline,
            w_edge=self.wedge,
            max_num_iter=self.n_iter,
            boundary_condition=self.boundary_condition,
        )

        # skimage doesn't expose per-step history; simulate by running in chunks
        chunk = max(1, self.n_iter // 10)
        temp = init_snake.copy()
        history.append(temp.copy())
        for _ in range(10):
            temp = active_contour(
                img_smooth,
                temp,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                w_line=self.wline,
                w_edge=self.wedge,
                max_num_iter=chunk,
                boundary_condition=self.boundary_condition,
            )
            history.append(temp.copy())

        return snake, history

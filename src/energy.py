import numpy as np
from scipy.ndimage import gaussian_filter


def compute_gradient_magnitude(image, sigma=1.0):
    """Normalized gradient magnitude map, used for adaptive weight modulation."""
    img = gaussian_filter(image.astype(float), sigma=max(sigma, 0.1))
    gy, gx = np.gradient(img)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    denom = gmag.max()
    return gmag / denom if denom > 0 else gmag


def compute_external_energy(image, sigma=1.0, wline=0.0, wedge=1.0):
    """
    External energy map E_ext = -wedge*|∇(G_σ*I)|² - wline*G_σ*I.
    Lower values attract the contour.
    """
    img = gaussian_filter(image.astype(float), sigma=max(sigma, 0.1))
    gy, gx = np.gradient(img)
    edge_term = -wedge * (gx ** 2 + gy ** 2)
    line_term = -wline * img
    return edge_term + line_term


def compute_external_forces(image, sigma=1.0, wline=0.0, wedge=1.0):
    """
    External force field F_ext = -∇E_ext (points toward edge features).
    Returns (fy, fx) arrays matching (row, col) convention.
    """
    energy = compute_external_energy(image, sigma, wline, wedge)
    fy, fx = np.gradient(energy)
    # Negate: forces point toward energy minima
    return -fy, -fx

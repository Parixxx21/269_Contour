import numpy as np
from skimage import io, color
from skimage.transform import resize


def load_grayscale(path, target_shape=None):
    """Load any image file as a float grayscale array in [0, 1]."""
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(float)
    img -= img.min()
    img /= img.max() + 1e-8
    if target_shape is not None:
        img = resize(img, target_shape, anti_aliasing=True)
    return img


def add_gaussian_noise(image, sigma=0.1, seed=None):
    """Add zero-mean Gaussian noise; result clipped to [0, 1]."""
    rng = np.random.default_rng(seed)
    return np.clip(image + rng.normal(0, sigma, image.shape), 0, 1)


def add_salt_pepper(image, density=0.05, seed=None):
    """Add salt-and-pepper noise at given pixel density."""
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    mask = rng.random(image.shape)
    noisy[mask < density / 2] = 0.0
    noisy[mask > 1 - density / 2] = 1.0
    return noisy


def reduce_contrast(image, factor=0.3):
    """
    Pull pixel values toward the image mean by *factor* in [0, 1].
    factor=1 leaves image unchanged; factor=0 produces a flat gray image.
    """
    mean = image.mean()
    return np.clip(mean + factor * (image - mean), 0, 1)

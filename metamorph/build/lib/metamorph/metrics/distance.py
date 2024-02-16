import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance

# Distance measurements
# =====================
# JS:          Jensen-Shannon divergence
# Wasserstein: Wasserstein distance
# JS divergence fails to provide a useful gradient 
#   when the distributions are supported on non-overlapping domains


def cov2d_var(mat: torch.Tensor or np.ndarray) -> np.ndarray:
    """
    Calculate the variance of an Cov2D output.
    """
    # mat: batch_size x n_channels x height x width
    mat = mat.cpu().detach().numpy()
    batch_size, n_channels, height, width = mat.shape

    # Flatten the height and width dimensions
    mat = mat.reshape(batch_size, n_channels, -1)

    # Compute variance of each pixel across all channels
    # mat_var: batch_size x n_pixel
    mat_var = np.var(mat, axis=1)

    return mat_var


def cov2d_jensenshannon(mat1: torch.Tensor or np.ndarray,
                        mat2: torch.Tensor or np.ndarray) -> float:
    """
    Calculate the Jensen-Shannon divergence of two Cov2D outputs.
    """
    # Compute variances of each output
    mat1_var = cov2d_var(mat1)
    mat2_var = cov2d_var(mat2)

    # Sanity check
    batch_size1, n_pixel1 = mat1_var.shape
    batch_size2, n_pixel2 = mat2_var.shape
    assert batch_size1 == batch_size2, "Batch size mismatch"
    assert n_pixel1 == n_pixel2,       "Pixel size mismatch"

    # Compute JS divergence on each batch then take compute the mean
    return np.mean(jensenshannon(
        mat1_var[i], mat2_var[i]
    ) for i in range(batch_size1))


def cov2d_wasserstein(mat1: torch.Tensor or np.ndarray,
                        mat2: torch.Tensor or np.ndarray) -> float:
    """
    Calculate the Jensen-Shannon divergence of two Cov2D outputs.
    """
    # Compute variances of each output
    mat1_var = cov2d_var(mat1)
    mat2_var = cov2d_var(mat2)

    # Sanity check
    batch_size1, n_pixel1 = mat1_var.shape
    batch_size2, n_pixel2 = mat2_var.shape
    assert batch_size1 == batch_size2, "Batch size mismatch"
    assert n_pixel1 == n_pixel2,       "Pixel size mismatch"

    # Compute JS divergence on each batch then take compute the mean
    return np.mean(wasserstein_distance(
        mat1_var[i], mat2_var[i]
    ) for i in range(batch_size1))

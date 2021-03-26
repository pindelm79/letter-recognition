"""This module contains fast helper functions for the nn package, wrapped with numba."""
from numba import njit
import numpy as np

# ---Conv2d helper functions---
@njit
def calculate_input_gradient(
    dout: np.ndarray, in_array: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Calculates the gradient (w.r.t. the output) of the input.

    Parameters
    ----------
    dout : np.ndarray
        "Upstream" gradients. Shape: (N, C_out, H_out, W_out).
    in_array : np.ndarray
        Last input to the layer. Shape: (N, C_in, H_in, W_in).
    weight : np.ndarray
        Kernel of the convolution. Shape: (C_out, C_in, H_kernel, W_kernel)

    Returns
    -------
    np.ndarray
        Input gradient. Shape: (N, C_in, H_in, W_in).
    """
    dx = np.zeros(in_array.shape)

    for n in range(dout.shape[0]):  # N
        for f in range(dout.shape[1]):  # output channels
            for c in range(in_array.shape[1]):  # input channels
                for i in range(dout.shape[2]):  # image height
                    for j in range(dout.shape[3]):  # image width
                        # ^Using dout's shape because it is already constrained
                        for k in range(weight.shape[2]):  # kernel height
                            for l in range(weight.shape[3]):  # kernel width
                                dx[n, c, i + k, j + l] += (
                                    weight[f, c, k, l] * dout[n, f, i, j]
                                )

    return dx


@njit
def calculate_weight_gradient(
    dout: np.ndarray, in_array: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """Calculates the gradient (w.r.t. the output) of the weights.

    Parameters
    ----------
    dout : np.ndarray
        "Upstream" gradients. Shape: (N, C_out, H_out, W_out).
    in_array : np.ndarray
        Last input to the layer. Shape: (N, C_in, H_in, W_in).
    weight : np.ndarray
        Kernel of the convolution. Shape: (C_out, C_in, H_kernel, W_kernel)

    Returns
    -------
    np.ndarray
        Weight gradient. Shape: (C_out, C_in, H_kernel, W_kernel).
    """
    dW = np.zeros(weight.shape)

    for n in range(dout.shape[0]):  # N
        for f in range(dout.shape[1]):  # output channels
            for c in range(in_array.shape[1]):  # input channels
                for i in range(dout.shape[2]):  # image height
                    for j in range(dout.shape[3]):  # image width
                        # ^Using dout's shape because it is already constrained
                        for k in range(weight.shape[2]):  # kernel height
                            for l in range(weight.shape[3]):  # kernel width
                                dW[f, c, k, l] += (
                                    in_array[n, c, i + k, j + l] * dout[n, f, i, j]
                                )

    return dW

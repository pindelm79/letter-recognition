"""This module contains fast helper functions for the nn package, usually wrapped with numba."""
from typing import Tuple, Union
from _pytest.fixtures import wrap_function_to_error_out_if_called_directly

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


# ---MaxPool2d helper functions---
@njit
def maxpool2d_forward(
    in_array: np.ndarray,
    out: np.ndarray,
    stride: Tuple[int, int],
    kernel_size: Tuple[int, int],
    padding: Tuple[int, int],
    ceil_mode: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Does maxpooling over an input.

    Parameters
    ----------
    in_array : np.ndarray
        The input array to maxpool over. Shape: (N, C, H_in, W_in).
    out : np.ndarray
        The output array, uninitialized. Shape: (N, C, H_out, W_out).
    stride : tuple of 2 ints
        Stride of the maxpool.
    kernel_size : tuple of 2 ints
        Kernel size of the maxpool.
    padding: tuple of 2 ints
        Padding of the input.
    ceil_mode: bool
        Whether to use ceiling mode.

    Returns
    -------
    ndarray or tuple of 2 ndarrays
        Output of the maxpool.
        If return_indices=True, also returns an array of (flat) indices of max values.
    """
    # TODO: write a numba-friendly version of the function - ravel/unravel are problematic

    max_indices = np.empty(out.shape, dtype="int64")

    for N in range(out.shape[0]):
        for C in range(out.shape[1]):
            for h in range(out.shape[2]):
                for w in range(out.shape[3]):
                    # Implicit padding
                    h_slice_start = max(stride[0] * h - padding[0], 0)
                    h_slice_end = h_slice_start + kernel_size[0]
                    if stride[0] * h - padding[0] < 0:
                        # If start would dip below 0, offset the end by the diff. Otherwise, the start is already offset.
                        h_slice_end -= abs(stride[0] * h - padding[0])

                    w_slice_start = max(stride[1] * w - padding[1], 0)
                    w_slice_end = w_slice_start + kernel_size[1]
                    if stride[1] * w - padding[1] < 0:
                        w_slice_end -= abs(stride[1] * w - padding[1])

                    # Check if OOB
                    if h_slice_start >= in_array.shape[2]:
                        continue
                    if h_slice_end > in_array.shape[2]:
                        if (
                            h_slice_end > in_array.shape[2] + padding[0]
                            and not ceil_mode
                        ):  # Implicit down-padding
                            continue
                        h_slice_end = in_array.shape[2]
                    if w_slice_start >= in_array.shape[3]:
                        continue
                    if w_slice_end > in_array.shape[3]:
                        if (
                            w_slice_end > in_array.shape[3] + padding[1]
                            and not ceil_mode
                        ):  # Implicit right-padding
                            continue
                        w_slice_end = in_array.shape[3]

                    # Get flat and non-flat slice index of max
                    index_slice_flat = np.argmax(
                        in_array[
                            N,
                            C,
                            h_slice_start:h_slice_end,
                            w_slice_start:w_slice_end,
                        ]
                    )
                    # Unravel index (1D to 2D)
                    slice_shape = (
                        h_slice_end - h_slice_start,
                        w_slice_end - w_slice_start,
                    )
                    index_slice = (
                        index_slice_flat // slice_shape[1],
                        index_slice_flat % slice_shape[1],
                    )

                    # Calculate 2d flat and non-flat index from the slice
                    index_2d = (
                        h_slice_start + index_slice[0],
                        w_slice_start + index_slice[1],
                    )
                    shape_2d = (in_array.shape[2], in_array.shape[3])
                    index_2d_flat = index_2d[0] * shape_2d[1] + index_2d[1]

                    # Assign the value and index
                    out[N, C, h, w] = in_array[N, C, index_2d[0], index_2d[1]]
                    max_indices[N, C, h, w] = index_2d_flat

    return out, max_indices


@njit
def maxpool2d_backward(
    dout: np.ndarray,
    in_array: np.ndarray,
    max_indices: Union[np.ndarray, None],
) -> np.ndarray:
    """Return the gradient of the output w.r.t. the maxpool input.

    Parameters
    ----------
    dout : np.ndarray
        "Upstream" gradient. Shape: (N, C, H_out, W_out).
    in_array : np.ndarray
        Previous input. Shape: (N, C, H_in, W_in).
    max_indices : np.ndarray or None
        If not None, an array of max indices in the in_array. As of now, assumes not None.

    Returns
    -------
    np.ndarray
        Gradient of the output w.r.t. the maxpool input. Shape: (N, C, H_in, W_in).
    """
    din = np.zeros_like(in_array)

    for N in range(dout.shape[0]):
        for C in range(dout.shape[1]):
            for H in range(dout.shape[2]):
                for W in range(dout.shape[3]):
                    max_flat_index = max_indices[N, C, H, W]
                    din[N, C].flat[max_flat_index] = dout[N, C, H, W]

    return din


# ---Other helper functions---
def pad4d(in_array: np.ndarray, padding: Tuple[int, int]) -> np.ndarray:
    """Pads 3rd and 4th dims of a 4D array with zeros.

    Parameters
    ----------
    in_array : np.ndarray
        Array to be padded.
    padding : tuple of 2 ints
        Zero padding to be applied in 3rd and 4th dims respectively.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    pad_array = np.zeros(
        (
            in_array.shape[0],
            in_array.shape[1],
            in_array.shape[2] + 2 * padding[0],
            in_array.shape[3] + 2 * padding[1],
        )
    )

    for i in range(in_array.shape[0]):
        for j in range(in_array.shape[1]):
            pad_array[i, j] = np.pad(
                in_array[i, j], ((padding[0], padding[0]), (padding[1], padding[1]))
            )

    return pad_array
"""This module contains layers which can be added into a model."""

from abc import ABC, abstractmethod
from math import ceil, floor, sqrt
from typing import Tuple, Union

import numpy as np
from scipy import signal

from letter_recognition import RNG
import letter_recognition.nn.ffunctions as fast


class _Layer(ABC):
    """An abstract class for defining layers/operations of a model."""

    @abstractmethod
    def forward(self, in_array: np.ndarray):
        """An abstract method for defining a forward pass through the layer."""
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray, in_array: np.ndarray):
        """An abstract method for defining a backward pass through the layer."""
        pass


class Conv2d(_Layer):
    """Defines a 2D convolution layer.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input.
    out_channels : int
        Number of channels in the output.
    kernel_size : int or tuple of 2 ints, optional
        Size of the convolving kernel. If a single int - it is the value for height and width.
        If a tuple of 2 ints - first is used for height, second for width.
    stride : int or tuple of 2 ints, optional
        Step size of the convolution. As of now, it can only be (1, 1).
    padding : int or tuple of 2 ints, optional
        Zero-padding added to all sides of the input. If a single int - it is the value for height
        and width. If a tuple of 2 ints - first is used for height, second for width. By default 0.
    bias : bool, optional
        If True, adds a leanable bias to the output. By default True.

    Attributes
    ----------
    in_channels
    out_channels
    kernel_size
    stride
    padding
    weight : np.ndarray
        Learnable kernel of shape (out_channels, in_channels, kernel_height, kernel_width).
    bias : np.ndarray
        Optional learnable additive bias of shape (out_channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride != 1 and stride != (1, 1):
            raise RuntimeError("Strides different than 1 are currently not supported.")
        self.stride = (1, 1)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self._use_bias = bias

        k = 1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.weight = RNG.uniform(
            -sqrt(k),
            sqrt(k),
            (
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            ),
        )
        if self._use_bias:
            self.bias = RNG.uniform(-sqrt(k), sqrt(k), self.out_channels)
        else:
            self.bias = np.zeros(self.out_channels)

    def forward(self, in_array: np.ndarray) -> np.ndarray:
        """Applies the specified 2D convolution over an input.

        Parameters
        ----------
        in_array : np.ndarray
            The input to apply the convolution to. Shape (N, C_in, H_in, W_in).

        Returns
        -------
        np.ndarray
            The output of the convolution. Shape: (N, C_out, H_out, W_out).

        Notes
        -----
        Formula:
        .. math::
            out[N_i, C_out_j] = bias[C_out_j] + sum(from k=0 to C_in-1) of:
            cross-correlate(kernel(C_out_j, k), input(N_i, k))
        """
        output = np.empty(self._calculate_output_shape(in_array.shape))

        if self.padding != (0, 0):
            in_array = fast.pad4d(in_array, self.padding)

        for n in range(output.shape[0]):  # N
            for f in range(output.shape[1]):  # output channels
                cross_correlation_sum = 0
                for c in range(self.in_channels):  # input channels
                    cross_correlation_sum += signal.correlate2d(
                        in_array[n, c], self.weight[f, c], mode="valid"
                    )
                output[n, f] = self.bias[f] + cross_correlation_sum

        return output

    def backward(
        self, dout: np.ndarray, in_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Does backpropagation through the conv2d layer.

        Parameters
        ----------
        dout : np.ndarray
            "Upstream" gradients. Shape: (N, C_out, H_out, W_out).
        in_array : np.ndarray
            Last input to the layer. Shape: (N, C_in, H_in, W_in).

        Returns
        -------
        tuple of 3 numpy arrays:
            dx - Shape: (N, C_in, H, W),
            dW - Shape: (C_out, C_in, kernel_H, kernel_W)
            db - Shape: (C_out).

        Notes
        -----
        Assumes stride = 1.
        """
        if self.padding != (0, 0):
            in_array = fast.pad4d(in_array, self.padding)

        dx = fast.calculate_input_gradient(dout, in_array, self.weight)
        # Remove padding from output
        new_H = dx.shape[2] - self.padding[0]
        new_W = dx.shape[3] - self.padding[1]
        dx = dx[:, :, self.padding[0]: new_H, self.padding[1]: new_W]

        dW = fast.calculate_weight_gradient(dout, in_array, self.weight)

        # Only calculate db if using bias, otherwise just zeros.
        if self._use_bias:
            db = np.sum(dout, axis=(0, 2, 3))
        else:
            db = np.zeros(self.out_channels)

        return dx, dW, db

    # ---Private helper functions.---
    def _calculate_output_shape(
        self, input_shape: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Calculates the shape of the output of the convolution.

        Parameters
        ----------
        input_size : tuple of 4 ints
            Shape of the input.

        Returns
        -------
        tuple of 4 ints
            Shape of the convolution output.
        """
        out_height = (
            (input_shape[2] + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1)
            / self.stride[0]
        ) + 1
        out_width = (
            (input_shape[3] + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1)
            / self.stride[1]
        ) + 1
        return (input_shape[0], self.out_channels, int(out_height), int(out_width))


class Linear(_Layer):
    """Defines a fully connected (linear) layer.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        If True, adds a learnable bias to the output. By default True.

    Attributes
    ----------
    in_features
    out_features
    weight : np.ndarray
        Learnable weights of shape (out_features, in_features)
    bias : np.ndarray
        Learnable bias of shape (out_features)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self._use_bias = bias

        # Weight and bias initialization (as in PyTorch)
        k = 1 / in_features
        self.weight = RNG.uniform(-sqrt(k), sqrt(k), (out_features, in_features))
        if self._use_bias:
            self.bias = RNG.uniform(-sqrt(k), sqrt(k), (out_features,))
        else:
            self.bias = np.zeros((out_features,))

    def forward(self, in_array: np.ndarray) -> np.ndarray:
        """Applies the specified linear transformation over an input.

        Parameters
        ----------
        in_array : np.ndarray
            Input to apply the transformation to. Shape: (N, in_features), where N = no. of samples.

        Returns
        -------
        np.ndarray
            Output of the transformation. Shape: (N, out_features), where N = no. of samples.

        Notes
        -----
        .. math:: y = x W^T + b
        """
        return in_array @ self.weight.transpose() + self.bias

    def backward(
        self, dout: np.ndarray, in_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Does backpropagation through the Linear layer.

        Parameters
        ----------
        dout : np.ndarray
            "Upstream" gradients. Shape: (N, out_features).
        in_array : np.ndarray
            Last input to the layer. Shape: (N, in_features).

        Returns
        -------
        tuple of 3 numpy arrays
            dx - Shape: (N, in_features),
            dW - Shape: (out_features, in_features)
            db - Shape: (out_features).
        """
        dx = dout @ self.weight
        dW = dout.transpose() @ in_array
        db = np.sum(dout, axis=0)

        return dx, dW, db


class MaxPool2d(_Layer):
    """Defines a 2D max pooling layer.

    Parameters
    ----------
    kernel_size : int or tuple of 2 ints
        The size of the window to take a max over.
    stride : None or int or tuple of 2 ints, optional
        The stride of the window. If None it is equal to kernel_size.
        As of now, it can only be None or equal to kernel_size. By default None.
    padding : int or tuple of 2 ints, optional
        Implicit zero padding to be added on both sides. Must be equal to or smaller than half of
        kernel size. By default 0.
    dilation : int or tuple of 2 ints, optional
        A parameter that controls the stride of elements in the window.
        As of now, it can only be 1.
    return_indices : bool, optional
        If True, will return an array of (flat) indices of max values along with the usual
        output. By default True.
    ceil_mode : bool, optional
        If True, will use ceil instead of floor to compute the output shape. Sliding windows will
        be allowed to go off-bounds if they start within the left padding or the input.
        By default False.

    Notes
    -----
    In return_indices, the indices are calculated relatively, "inside" each sample and channel.
    That is, they are flat, but only in the context of the last 2 dimensions (H and W). Also,
    it doesn't count padding. This was done to ensure test compatibility with PyTorch.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[None, int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = True,
        ceil_mode: bool = False,
    ):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is not None and (stride != kernel_size or stride != self.kernel_size):
            raise RuntimeError(
                "Strides different than kernel size are currently not supported!"
            )
        self.stride = self.kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        if (self.padding[0] > (self.kernel_size[0] / 2)) or (
            self.padding[1] > (self.kernel_size[1] / 2)
        ):
            print(self.padding)
            raise RuntimeError(
                "Padding must be equal to or smaller than half of kernel size."
            )

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
        if self.dilation != (1, 1):
            raise RuntimeError(
                "Dilations different than 1 are currently not supported."
            )

        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(
        self, in_array: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Does maxpooling over the specified input.

        Parameters
        ----------
        in_array : np.ndarray
            The input array to maxpool over. Shape: (N, C, H_in, W_in).

        Returns
        -------
        ndarray or tuple of 2 ndarrays
            Returns the output of maxpooling. Shape: (N, C, H_out, W_out).
            If return_indices=True, also returns an array of (flat) indices of max values.
        """
        out = np.empty(self._calculate_output_shape(in_array.shape))

        out, max_indices = fast.maxpool2d_forward(
            in_array,
            out,
            self.stride,
            self.kernel_size,
            self.padding,
            self.ceil_mode,
        )

        if self.return_indices:
            return out, max_indices
        return out

    def backward(
        self,
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
        return fast.maxpool2d_backward(dout, in_array, max_indices)

    def _calculate_output_shape(
        self, in_shape: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Calculates the output shape.

        Parameters
        ----------
        input_shape : tuple of 4 ints
            Shape of the input (N, C, H_in, W_in).

        Returns
        -------
        tuple of 4 ints
            Shape of the output (N, C, H_out, W_out).
        """
        if not self.ceil_mode:
            H_out = floor(
                (
                    (
                        in_shape[2]
                        + 2 * self.padding[0]
                        - self.dilation[0] * (self.kernel_size[0] - 1)
                        - 1
                    )
                    / self.stride[0]
                )
                + 1
            )
            W_out = floor(
                (
                    (
                        in_shape[3]
                        + 2 * self.padding[1]
                        - self.dilation[1] * (self.kernel_size[1] - 1)
                        - 1
                    )
                    / self.stride[1]
                )
                + 1
            )
        else:
            H_out = ceil(
                (
                    (
                        in_shape[2]
                        + self.padding[0]
                        - self.dilation[0] * (self.kernel_size[0] - 1)
                        - 1
                    )
                    / self.stride[0]
                )
                + 1
            )
            W_out = ceil(
                (
                    (
                        in_shape[3]
                        + self.padding[1]
                        - self.dilation[1] * (self.kernel_size[1] - 1)
                        - 1
                    )
                    / self.stride[1]
                )
                + 1
            )
        return (in_shape[0], in_shape[1], H_out, W_out)

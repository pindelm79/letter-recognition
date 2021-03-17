"""This module contains layers which can be added into a model."""

from abc import ABC, abstractmethod
from typing import Tuple, Union
import math

import numpy as np
from scipy import signal


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
    """Defines a 2D convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ):
        """
        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the convolution layer.
            kernel_size: Size of the convolving kernel. If a single int - it is the value for
                height and width. If a tuple of 2 ints - first is used for height, second for
                width.
            stride: Stride ("step size") of the convolution.
            padding: Zero-padding added to both sides of the input. If a single int - specified
                number of zeros is added to all sides. If a tuple of 2 ints - first is used for
                adding zeros vertically, second horizontally.
            bias: If True, adds a learnable bias to the output.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride != 1 and stride != (1, 1):
            raise Warning("Strides different than 1 are currently not supported.")
        self.stride = (1, 1)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self._use_bias = bias

        k = 1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.weight = np.random.uniform(
            -math.sqrt(k),
            math.sqrt(k),
            (
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            ),
        )
        if self._use_bias:
            k = 1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])
            self.bias = np.random.uniform(
                -math.sqrt(k), math.sqrt(k), self.out_channels
            )
        else:
            self.bias = np.zeros(self.out_channels)

    def forward(self, in_array: np.ndarray) -> np.ndarray:
        """Applies the specified 2D convolution over an input.

        Formula:
            out(N_i, C_out_j) = bias(C_out_j) + Sum(from k=0 to C_in-1) of
            cross-correlate(kernel(C_out_j, k), input(N_i, k))

        Args:
            in_array: The input to be apply the convolution to. Shape: (N, C_in, H_in, W_in), where
                N = number of samples; C_in = number of channels in the input; H_in = height of
                the input; W_in = width of the input.

        Returns:
            The output of the convolution. Shape: (N, C_out, H_out, W_out), where N = number of
                samples; C_out = number of channels in the output; H_out = height of the output,
                calculated from the input and parameters; W_out = width of the output, calculated
                from the input and parameters.
        """
        output = np.empty(self._calculate_output_shape(in_array.shape))

        if self.padding != (0, 0):
            in_array = self._pad(in_array, self.padding)

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

        Assumes stride = 1.

        Args:
            dout: "Upstream" gradients. Shape: (N, C_out, H_out, W_out).
            in_array: Last input to the layer. Shape: (N, C_in, H_in, W_in).

        Returns:
            A tuple of:
                dx - Shape: (N, C_in, H, W),
                dW - Shape: (C_out, C_in, kernel_H, kernel_W)
                db - Shape: (C_out).
        """
        dx = self._calculate_input_gradient(dout, in_array)

        dW = self._calculate_weight_gradient(dout, in_array)

        # Only calculate db if using bias, otherwise just zeros.
        if self._use_bias:
            db = np.sum(dout, axis=(0, 2, 3))
        else:
            db = np.zeros(self.out_channels)

        return dx, dW, db

    # ---Private helper functions.---
    def _calculate_weight_gradient(
        self, dout: np.ndarray, in_array: np.ndarray
    ) -> np.ndarray:
        """Calculates the gradient (w.r.t. the output) of the weights (kernel).

        Args:
            dout: "Upstream" gradients. Shape: (N, C_out, H_out, W_out).
            in_array: Last input to the layer. Shape: (N, C_in, H_in, W_in).

        Returns:
            Array of weight gradients. Shape: (C_out, C_in, kernel_H, kernel_W).
        """
        dW = np.zeros(self.weight.shape)

        if self.padding != (0, 0):
            in_array = self._pad(in_array, self.padding)

        for n in range(dout.shape[0]):  # N
            for f in range(dout.shape[1]):  # output channels
                for c in range(self.in_channels):  # input channels
                    for i in range(dout.shape[2]):  # image width
                        for j in range(dout.shape[3]):  # image height
                            # ^Using dout's shape because it is already constrained
                            for k in range(self.kernel_size[0]):  # kernel width
                                for l in range(self.kernel_size[1]):  # kernel height
                                    dW[f, c, k, l] += (
                                        in_array[n, c, i + k, j + l] * dout[n, f, i, j]
                                    )

        return dW

    def _calculate_input_gradient(
        self, dout: np.ndarray, in_array: np.ndarray
    ) -> np.ndarray:
        """Calculates the gradient (w.r.t. the output) of the input.

        Args:
            dout: "Upstream" gradients. Shape: (N, C_out, H_out, W_out).
            in_array: Last input to the layer. Shape: (N, C_in, H_in, W_in).

        Returns:
            Array of input gradients. Shape: (N, C_in, H_in, W_in).
        """
        if self.padding != (0, 0):
            in_array = self._pad(in_array, self.padding)

        dx = np.zeros(in_array.shape)

        for n in range(dout.shape[0]):  # N
            for f in range(dout.shape[1]):  # output channels
                for c in range(self.in_channels):  # input channels
                    for i in range(dout.shape[2]):  # image width
                        for j in range(dout.shape[3]):  # image height
                            # ^Using dout's shape because it is already constrained
                            for k in range(self.kernel_size[0]):  # kernel width
                                for l in range(self.kernel_size[1]):  # kernel height
                                    dx[n, c, i + k, j + l] += (
                                        self.weight[f, c, k, l] * dout[n, f, i, j]
                                    )

        # Return dx with no padding
        new_H = dx.shape[2] - self.padding[0]
        new_W = dx.shape[3] - self.padding[1]
        return dx[:, :, self.padding[0] : new_H, self.padding[1] : new_W]

    def _calculate_output_shape(
        self, input_shape: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Helper function to calculate the shape of the output of the convolution.

        Args:
            input_size: Shape of the input.

        Returns:
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

    def _pad(self, in_array: np.ndarray, padding: Tuple[int, int]) -> np.ndarray:
        """Applies padding to the 4D input array.

        Args:
            in_array: Input array. Shape: (N, C_in, H_in, W_in)
            padding: Padding to apply to each dim.

        Returns:
            Padded input array.
        """
        padded_array = np.zeros(
            (
                in_array.shape[0],
                in_array.shape[1],
                in_array.shape[2] + 2 * padding[0],
                in_array.shape[3] + 2 * padding[1],
            )
        )
        for n in range(in_array.shape[0]):  # N
            for c in range(in_array.shape[1]):  # input channels
                padded_array[n, c] = np.pad(
                    in_array[n, c],
                    (
                        (self.padding[0], self.padding[0]),
                        (self.padding[1], self.padding[1]),
                    ),
                )
        return padded_array


class Linear(_Layer):
    """Defines a fully connected (linear) layer.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        If True, adds a learnable bias to the output.

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
        self.weight = np.random.uniform(
            -math.sqrt(k), math.sqrt(k), (out_features, in_features)
        )
        if self._use_bias:
            self.bias = np.random.uniform(-math.sqrt(k), math.sqrt(k), (out_features,))
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
        print(dout.shape)
        print(in_array.shape)
        dx = dout @ self.weight
        dW = dout.transpose() @ in_array
        db = np.sum(dout, axis=0)

        return dx, dW, db


if __name__ == "__main__":
    print("ok")
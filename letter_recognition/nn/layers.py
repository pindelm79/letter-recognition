"""This module contains layers which can be added into a model."""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from scipy import signal


class Layer(ABC):
    """An abstract class for defining layers/operations of a model."""

    @abstractmethod
    def forward(self, in_array: np.ndarray):
        """An abstract method for defining a forward pass through the layer."""
        pass

    @abstractmethod
    def backward(self):
        """An abstract method for defining a backward pass through the layer."""
        pass


class Conv2d(Layer):
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
        self.initialize_kernel()

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.use_bias = bias
        self.initialize_bias()

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
        output = np.empty(self.calculate_output_shape(in_array.shape))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                cross_correlation_sum = 0
                for k in range(self.in_channels):
                    padded_array = np.pad(
                        in_array[i, k],
                        (
                            (self.padding[0], self.padding[0]),
                            (self.padding[1], self.padding[1]),
                        ),
                    )
                    cross_correlation_sum += self.cross_correlate2d(
                        padded_array, self.kernel[j, k]
                    )
                output[i, j] = self.bias[j] + cross_correlation_sum

        return output

    def backward(self):
        raise NotImplementedError

    def cross_correlate2d(
        self, in1: np.ndarray, in2: np.ndarray, stride: Tuple[int, int] = (1, 1)
    ) -> np.ndarray:
        """Performs a cross-correlation on the given 2 arrays, with given stride.

        TODO: implement my own.

        Args:
            in1: 1st 2D array.
            in2: 2nd 2D array.
            stride: "Step size" of the convolution in each dimension.

        Returns:
            np.ndarray: [description]
        """
        return signal.correlate2d(in1, in2, mode="valid")

    def calculate_output_shape(
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

    def initialize_kernel(self):
        """Initializes the kernel with weights of 1.

        Shape: (out_channels, in_channels, kernel_size[0], kernel_size[1]).

        TODO: initialize with sth else than ones
        """
        self.kernel = np.ones(
            (
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )

    def initialize_bias(self):
        """Initializes the biases with zeros.

        Shape: (out_channels)

        TODO: initialize with sth else than zeros
        """
        self.bias = np.zeros(self.out_channels)
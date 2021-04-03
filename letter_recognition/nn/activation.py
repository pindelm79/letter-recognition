"""This module contains activation functions."""

from abc import ABC, abstractmethod

import numpy as np


class _Activation(ABC):
    """An abstract class for defining activation functions of a model."""

    @abstractmethod
    def forward(self, in_array: np.ndarray):
        """An abstract method for defining a forward pass of the activation."""
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray, in_array: np.ndarray):
        """An abstract method for defining a backward pass of the activation."""
        pass


class ReLU(_Activation):
    """Applies the rectified linear unit function (max(0, x))."""

    def forward(self, in_array: np.ndarray) -> np.ndarray:
        """Does the forward pass of ReLu.

        Parameters
        ----------
        in_array : np.ndarray
            Input array. Shape: (N, *)

        Returns
        -------
        np.ndarray
            Output array. Shape: (N, *), same as input.
        """
        return np.maximum(0, in_array)

    def backward(self, dout: np.ndarray, in_array: np.ndarray):
        """Does a backward pass of the ReLU function.

        Parameters
        ----------
        dout : np.ndarray
            "Upstream" gradients. Shape: (N, *)
        in_array : np.ndarray
            Previous input.

        Returns
        -------
        np.ndarray
            Gradient of the input.
        """
        din = np.zeros_like(in_array)
        din[in_array > 0] = 1

        return din * dout

"""This module contains loss functions to evaluate model performance."""

from abc import ABC, abstractmethod

import numpy as np


class _Loss(ABC):
    """Abstract class for defining loss functions."""

    @abstractmethod
    def calculate(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Abstact method for calculating the loss."""
        pass

    def backward(self, predicted: np.ndarray) -> np.ndarray:
        """Abstract method for calculating loss/input gradient."""
        pass


class MAE(_Loss):
    """Simple mean absolute error loss."""

    def calculate(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculates the mean absolute error loss.

        Parameters
        ----------
        in_array : np.ndarray
            Output of a model. Shape: (N).
        target : np.ndarray
            Target (true) values. Shape: (N).

        Returns
        -------
        np.ndarray
            Mean absolute error.
        """
        return np.abs(predicted - target)

    def backward(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the input to the loss.

        Parameters
        ----------
        predicted : np.ndarray
            Input to the loss function. Shape: (N).
        target : np.ndarray
            Target (true) values. Shape: (N).

        Returns
        -------
        np.ndarray
            Gradient of the input.
        """
        gradient = np.zeros_like(predicted)
        for i in range(predicted.shape[0]):  # number of samples
            if predicted[i] > target[i]:
                gradient[i] = 1
            elif predicted[i] < target[i]:
                gradient[i] = -1
            else:
                gradient[i] = 0

        return gradient

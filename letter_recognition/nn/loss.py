"""This module contains loss functions to evaluate model performance."""

from abc import ABC, abstractmethod
import warnings

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
    """Simple mean absolute error loss.

    Parameters
    ----------
    reduction : 'none' | 'mean' | 'sum
        'none': no reduction will be applied; 'mean': the sum of the output will be divided by its size;
        'sum': the output will be summed.
    """

    def __init__(self, reduction: str = "mean"):
        reduction = reduction.lower()
        if reduction not in ["none", "mean", "sum"]:
            warnings.warn("Wrong reduction mode! Setting to default 'mean'.")
            reduction = "mean"
        self.reduction = reduction

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
        if self.reduction == "none":
            return np.abs(predicted - target)
        elif self.reduction == "mean":
            return np.mean(np.abs(predicted - target))
        elif self.reduction == "sum":
            return np.sum(np.abs(predicted - target))

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

        Notes
        -----
            If the reduction was 'sum' or no reduction, the gradient factor is 1.
            If 'mean', it is 1/N.
        """
        gradient = np.zeros_like(predicted)

        grad_factor = 1.0
        if self.reduction == "mean":
            grad_factor = 1 / gradient.shape[0]  # 1 / N

        for i in range(predicted.shape[0]):  # N
            if predicted[i] > target[i]:
                gradient[i] = grad_factor
            elif predicted[i] < target[i]:
                gradient[i] = -grad_factor
            else:
                gradient[i] = 0

        return gradient

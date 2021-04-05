"""This module contains loss functions to evaluate model performance."""

from abc import ABC, abstractmethod
from typing import Union

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
    reduction : 'none' | 'mean' | 'sum', optional
        'none': no reduction will be applied, 'mean': the sum of the output will be divided by its size,
        'sum': the output will be summed. By default 'mean'.
    """

    def __init__(self, reduction: str = "mean"):
        reduction = reduction.lower()
        if reduction not in ["none", "mean", "sum"]:
            raise RuntimeError("Wrong reduction mode!")
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

        return gradient


class CrossEntropy(_Loss):
    """Combines log(Softmax(x)) with negative log likelihood.

    In a multi-class classification problem, it describes "how correct/wrong" the probability
    of the correct class is.

    Parameters
    ----------
    weight : ndarray or None, optional
        If not None, expects an array of size C (number of classes) with weight given to each
        class. Useful with unbalanced classes. If None, it assumes all weights are 1.
        By default None.
    reduction : 'none' | 'mean' | 'sum', optional
        'none': no reduction will be applied, 'mean': the sum of the output will be divided by its size,
        'sum': the output will be summed. By default 'mean'.
    """

    def __init__(self, weight: Union[np.ndarray, None] = None, reduction: str = "mean"):
        self.weight = weight
        reduction = reduction.lower()
        if reduction not in ["none", "mean", "sum"]:
            raise RuntimeError("Wrong reduction mode!")
        self.reduction = reduction

    def calculate(self, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
        return super().calculate(predicted, target)

    def backward(self, predicted: np.ndarray) -> np.ndarray:
        return super().backward(predicted)
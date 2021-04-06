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

    def calculate(
        self, predicted: np.ndarray, target: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Calculates the mean absolute error loss.

        Parameters
        ----------
        predicted : np.ndarray
            Output of a model. Shape: (N).
        target : np.ndarray
            Target (true) values. Shape: (N).

        Returns
        -------
        float or np.ndarray
            If no reduction, returns a loss for each N. Otherwise, returns the aggregated loss.
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

    def calculate(
        self, predicted: np.ndarray, true_classes: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Calculates the cross entropy loss.

        Parameters
        ----------
        predicted : np.ndarray
            Values predicted by the model. Shape: (N, C).
        true_classes : ndarray of ints (each from 0 to C-1).
            Array of index of the correct classes. Shape: (N,).

        Returns
        -------
        float or np.ndarray
            If no reduction, returns a loss for each N. Otherwise, returns the aggregated loss.
        """
        if self.weight is None:
            self.weight = np.ones(predicted.shape[1])  # C

        loss_array = np.empty(predicted.shape[0])  # N
        weight_sum = 0.0

        for N in range(loss_array.shape[0]):
            true_class = int(true_classes[N])
            loss_array[N] = -predicted[N, true_class] + np.log(
                np.sum(np.exp(predicted[N]))
            )
            loss_array[N] *= self.weight[true_class]
            weight_sum += self.weight[true_class]

        if self.reduction == "none":
            return loss_array
        elif self.reduction == "sum":
            return np.sum(loss_array)
        elif self.reduction == "mean":
            return np.sum(loss_array) / weight_sum

    def backward(self, predicted: np.ndarray, true_classes: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the input to the loss.

        Parameters
        ----------
        predicted : np.ndarray
            Values predicted by the model. Shape: (N, C).
        true_classes : ndarray of ints (each from 0 to C-1).
            Array of index of the correct classes. Shape: (N,).

        Returns
        -------
        np.ndarray
            Gradient of the input.
        """
        if self.weight is None:
            self.weight = np.ones(predicted.shape[1])  # C

        gradient = np.zeros_like(predicted)
        weight_sum = 0.0

        for N in range(predicted.shape[0]):  # N
            true_class = int(true_classes[N])
            for C in range(predicted.shape[1]):  # C
                if C == true_class:
                    gradient[N, C] = (
                        np.exp(predicted[N, C]) / np.sum(np.exp(predicted[N]))
                    ) - 1
                else:
                    gradient[N, C] = np.exp(predicted[N, C]) / np.sum(
                        np.exp(predicted[N])
                    )
                gradient[N, C] *= self.weight[true_class]
            weight_sum += self.weight[true_class]

        if self.reduction == "mean":
            return gradient / weight_sum
        return gradient
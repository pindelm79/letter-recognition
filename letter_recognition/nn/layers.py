"""This module contains layers which can be added into a model."""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np


class Layer(ABC):
    """An abstract class for defining layers of a model."""

    @abstractmethod
    def forward(self, input: np.ndarray):
        """An abstract method for defining a forward pass through the layer."""
        pass

    @abstractmethod
    def backward(self):
        """An abstract method for defining a backward pass through the layer."""
        pass
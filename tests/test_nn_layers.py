from typing import Tuple
import numpy as np
import pytest
import torch
from torch._C import dtype
import torch.nn

from letter_recognition.nn import layers


@pytest.fixture
def conv2d_custom():
    """Returns a custom Conv2d with in_channels=1, out_channels=1, kernel_size=3."""
    return layers.Conv2d(1, 1, 3)


@pytest.fixture
def conv2d_torch(conv2d_custom):
    """Returns a PyTorch Conv2d derived from custom Conv2d."""
    return torch.nn.Conv2d(
        conv2d_custom.in_channels,
        conv2d_custom.out_channels,
        conv2d_custom.kernel_size,
        bias=False,
    )


@pytest.mark.parametrize("input_dims", [(1, 1, 50, 100)])
def test_conv2d_calculate_output_shape(conv2d_custom, conv2d_torch, input_dims: Tuple):
    np_array = np.random.random_sample(input_dims)
    torch_tensor = torch.from_numpy(np_array).float()

    out_shape = conv2d_custom.calculate_output_shape(input_dims)

    assert out_shape == conv2d_torch(torch_tensor).size()
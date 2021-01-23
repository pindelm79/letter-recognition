from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from letter_recognition.nn import layers


@pytest.fixture
def conv2d_custom():
    """Returns a custom Conv2d with in_channels=1, out_channels=1, kernel_size=3."""
    return layers.Conv2d(1, 1, 3, bias=False)


@pytest.mark.parametrize("input_dims", [(1, 1, 50, 100)])
def test_conv2d_forward(conv2d_custom, input_dims: Tuple):
    in_array = np.random.random_sample(input_dims)
    in_tensor = torch.from_numpy(in_array).float()

    output = conv2d_custom.forward(in_array)

    kernel_tensor = torch.from_numpy(conv2d_custom.kernel).float()
    expected = F.conv2d(in_tensor, kernel_tensor)

    assert output.shape == expected.size()
    assert torch.allclose(torch.from_numpy(output).float(), expected)


@pytest.mark.parametrize("input_dims", [(1, 1, 50, 100)])
def test_conv2d_calculate_output_shape(conv2d_custom, input_dims: Tuple):
    in_array = np.random.random_sample(input_dims)
    in_tensor = torch.from_numpy(in_array).float()

    out_shape = conv2d_custom.calculate_output_shape(input_dims)

    kernel_tensor = torch.from_numpy(conv2d_custom.kernel).float()
    expected_shape = F.conv2d(in_tensor, kernel_tensor).size()

    assert out_shape == expected_shape
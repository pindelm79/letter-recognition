from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from letter_recognition.nn import layers


@pytest.mark.parametrize(
    "N, in_channels, in_H, in_W, out_channels, kernel_size, bias",
    [
        (1, 1, 50, 100, 1, 3, False),
        (20, 16, 50, 100, 33, 3, True),
        (4, 3, 28, 28, 10, (3, 5), True),
    ],
)
def test_conv2d_forward(N, in_channels, in_H, in_W, out_channels, kernel_size, bias):
    conv2d_custom = layers.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
    in_shape = (N, in_channels, in_H, in_W)
    in_array = np.random.random_sample(in_shape)
    in_tensor = torch.from_numpy(in_array).float()

    output = conv2d_custom.forward(in_array)

    kernel_tensor = torch.from_numpy(conv2d_custom.kernel).float()
    bias_tensor = torch.from_numpy(conv2d_custom.bias).float()
    expected = F.conv2d(in_tensor, kernel_tensor, bias=bias_tensor)

    assert output.shape == expected.size()
    assert torch.allclose(torch.from_numpy(output).float(), expected)


@pytest.mark.parametrize(
    "N, in_channels, in_H, in_W, out_channels, kernel_size, bias",
    [
        (1, 1, 50, 100, 1, 3, False),
        (20, 16, 50, 100, 33, 3, True),
        (4, 3, 28, 28, 10, (3, 5), True),
    ],
)
def test_conv2d_calculate_output_shape(
    N, in_channels, in_H, in_W, out_channels, kernel_size, bias
):
    conv2d_custom = layers.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
    in_shape = (N, in_channels, in_H, in_W)
    in_array = np.random.random_sample(in_shape)
    in_tensor = torch.from_numpy(in_array).float()

    out_shape = conv2d_custom.calculate_output_shape(in_shape)

    kernel_tensor = torch.from_numpy(conv2d_custom.kernel).float()
    expected_shape = F.conv2d(in_tensor, kernel_tensor).size()

    assert out_shape == expected_shape
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from letter_recognition.nn import layers


@pytest.mark.parametrize(
    "N, in_channels, in_H, in_W, out_channels, kernel_size, stride, padding, bias",
    [
        (1, 1, 50, 100, 1, 3, 1, 0, False),
        (20, 16, 50, 100, 33, 3, 1, 2, True),
        (4, 3, 28, 28, 10, (3, 5), 1, (2, 5), True),
    ],
)
class TestConv2d:
    def test_conv2d_forward(
        self,
        N,
        in_channels,
        in_H,
        in_W,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
    ):
        conv2d_custom = layers.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        in_shape = (N, in_channels, in_H, in_W)
        in_array = np.random.random_sample(in_shape)
        in_tensor = torch.from_numpy(in_array).float()

        output = conv2d_custom.forward(in_array)

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        bias_tensor = torch.from_numpy(conv2d_custom.bias).float()
        expected = F.conv2d(
            in_tensor, weight_tensor, stride=stride, padding=padding, bias=bias_tensor
        )

        assert torch.allclose(torch.from_numpy(output).float(), expected)

    def test_conv2d_calculate_output_shape(
        self,
        N,
        in_channels,
        in_H,
        in_W,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
    ):
        conv2d_custom = layers.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        in_shape = (N, in_channels, in_H, in_W)
        in_array = np.random.random_sample(in_shape)
        in_tensor = torch.from_numpy(in_array).float()

        out_shape = conv2d_custom.calculate_output_shape(in_shape)

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        expected_shape = F.conv2d(
            in_tensor, weight_tensor, stride=stride, padding=padding
        ).size()

        assert out_shape == expected_shape
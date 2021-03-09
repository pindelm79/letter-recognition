from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn
import torch.nn.functional as F

from letter_recognition.nn import layers


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, in_H, in_W, kernel_size",
    [(1, 1, 1, 4, 4, 3), (50, 1, 9, 28, 28, 3), (69, 9, 17, 29, 13, (2, 5))],
)
class TestConv2d:
    @pytest.mark.parametrize("padding", [0, 2, (2, 1)])
    @pytest.mark.parametrize("bias", [False, True])
    def test_forward(
        self,
        batch_size,
        in_channels,
        in_H,
        in_W,
        out_channels,
        kernel_size,
        padding,
        bias,
    ):
        in_shape = (batch_size, in_channels, in_H, in_W)
        in_array = np.random.random_sample(in_shape)
        in_tensor = torch.from_numpy(in_array).float()

        conv2d_custom = layers.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=bias
        )
        out_custom = conv2d_custom.forward(in_array)

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        bias_tensor = torch.from_numpy(conv2d_custom.bias).float()
        out_torch = F.conv2d(
            in_tensor, weight_tensor, bias=bias_tensor, padding=padding
        )

        assert torch.allclose(torch.from_numpy(out_custom).float(), out_torch)

    @pytest.mark.parametrize("padding", [0, 2, (2, 1)])
    def test_calculate_output_shape(
        self, batch_size, in_channels, in_H, in_W, out_channels, kernel_size, padding
    ):
        in_shape = (batch_size, in_channels, in_H, in_W)
        in_array = np.random.random_sample(in_shape)
        in_tensor = torch.from_numpy(in_array).float()

        conv2d_custom = layers.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )
        out_shape_custom = conv2d_custom.calculate_output_shape(in_shape)

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        out_shape_torch = F.conv2d(in_tensor, weight_tensor, padding=padding).size()

        assert out_shape_custom == out_shape_torch

    def test_calculate_bias_gradient(
        self, batch_size, in_channels, in_H, in_W, out_channels, kernel_size
    ):
        in_shape = (batch_size, in_channels, in_H, in_W)
        in_array = np.random.random_sample(in_shape)
        in_tensor = torch.from_numpy(in_array).float()

        conv2d_custom = layers.Conv2d(in_channels, out_channels, kernel_size)
        # out_custom = conv2d_custom.forward(in_array)

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        bias_tensor = torch.from_numpy(conv2d_custom.bias).float()
        bias_tensor.requires_grad_(True)
        out_torch = F.conv2d(in_tensor, weight_tensor, bias=bias_tensor)

        out_torch.retain_grad()
        final = out_torch * 2
        final.sum().backward()
        bias_gradient_torch = bias_tensor.grad

        bias_gradient_custom = conv2d_custom.calculate_bias_gradient(
            out_torch.grad.numpy(), in_array
        )

        assert torch.allclose(
            torch.from_numpy(bias_gradient_custom).float(), bias_gradient_torch
        )
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn_torch
import torch.nn.functional as F

import letter_recognition.nn as nn_custom


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, in_H, in_W, kernel_size",
    [(1, 1, 1, 3, 3, 2), (4, 1, 9, 28, 28, 3), (10, 3, 10, 29, 13, (1, 4))],
)
@pytest.mark.parametrize("padding", [0, 2, (2, 1)])
@pytest.mark.parametrize("bias", [False, True])
class TestConv2d:
    def test_integration(
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
        # For now, just checking if there are no errors.
        in_shape = (batch_size, in_channels, in_H, in_W)
        in_array = np.random.randint(0, 256, in_shape).astype("float")
        conv2d = nn_custom.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=bias
        )
        out = conv2d.forward(in_array)
        gradient = np.full_like(out, 2)
        dx, dW, db = conv2d.backward(gradient, in_array)

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
        in_array = np.random.randint(0, 256, in_shape).astype("float")
        in_tensor = torch.from_numpy(in_array).float()

        conv2d_custom = nn_custom.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=bias
        )
        out_custom = conv2d_custom.forward(in_array)

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        bias_tensor = torch.from_numpy(conv2d_custom.bias).float()
        out_torch = F.conv2d(
            in_tensor, weight_tensor, bias=bias_tensor, padding=padding
        )

        assert out_custom.shape == out_torch.size()
        assert torch.allclose(
            torch.from_numpy(out_custom).float(), out_torch, atol=1e-4
        )

    def test_calculate_output_shape(
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
        in_array = np.random.randint(0, 256, in_shape).astype("float")
        in_tensor = torch.from_numpy(in_array).float()

        conv2d_custom = nn_custom.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )
        out_shape_custom = conv2d_custom.calculate_output_shape(in_shape)

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        out_shape_torch = F.conv2d(in_tensor, weight_tensor, padding=padding).size()

        assert out_shape_custom == out_shape_torch

    def test_calculate_bias_gradient(
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
        in_array = np.random.randint(0, 256, in_shape).astype("float")
        in_tensor = torch.from_numpy(in_array).float()

        conv2d_custom = nn_custom.Conv2d(in_channels, out_channels, kernel_size)

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        bias_tensor = torch.from_numpy(conv2d_custom.bias).float()
        bias_tensor.requires_grad_(True)
        out_torch = F.conv2d(in_tensor, weight_tensor, bias=bias_tensor)

        out_torch.retain_grad()
        final = out_torch * 2
        final.sum().backward()
        bias_gradient_torch = bias_tensor.grad

        bias_gradient_custom = conv2d_custom.calculate_bias_gradient(
            out_torch.grad.numpy()
        )

        assert bias_gradient_custom.shape == bias_gradient_torch.size()
        assert torch.allclose(
            torch.from_numpy(bias_gradient_custom).float(),
            bias_gradient_torch,
            atol=1e-4,
        )

    def test_calculate_weight_gradient(
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
        in_array = np.random.randint(0, 256, in_shape).astype("float")
        in_tensor = torch.from_numpy(in_array).float()

        conv2d_custom = nn_custom.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        weight_tensor.requires_grad_(True)
        out_torch = F.conv2d(in_tensor, weight_tensor, padding=padding)
        out_torch.retain_grad()

        final = out_torch * 2
        final.sum().backward()
        weight_gradient_torch = weight_tensor.grad

        weight_gradient_custom = conv2d_custom.calculate_weight_gradient(
            out_torch.grad.numpy(), in_array
        )

        assert weight_gradient_custom.shape == weight_gradient_torch.size()
        assert torch.allclose(
            torch.from_numpy(weight_gradient_custom).float(),
            weight_gradient_torch,
            atol=1e-4,
        )

    def test_calculate_input_gradient(
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
        in_array = np.random.randint(0, 256, in_shape).astype("float")
        in_tensor = torch.from_numpy(in_array).float()
        in_tensor.requires_grad_(True)

        conv2d_custom = nn_custom.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        out_torch = F.conv2d(in_tensor, weight_tensor, padding=padding)
        out_torch.retain_grad()

        final = out_torch * 2
        final.sum().backward()
        input_gradient_torch = in_tensor.grad

        input_gradient_custom = conv2d_custom.calculate_input_gradient(
            out_torch.grad.numpy(), in_array
        )

        assert input_gradient_custom.shape == input_gradient_torch.size()
        assert torch.allclose(
            torch.from_numpy(input_gradient_custom).float(),
            input_gradient_torch,
            atol=1e-4,
        )


@pytest.mark.parametrize(
    "batch_size, in_features, out_features", [(1, 3, 2), (4, 400, 100), (50, 20, 500)]
)
@pytest.mark.parametrize("bias", [False, True])
class TestLinear:
    def test_forward(self, batch_size, in_features, out_features, bias):
        in_shape = (batch_size, in_features)
        in_array = np.random.randint(0, 256, in_shape).astype("float")
        in_tensor = torch.from_numpy(in_array).float()

        linear_custom = nn_custom.Linear(in_features, out_features, bias)
        out_custom = linear_custom.forward(in_array)

        weight_tensor = torch.from_numpy(linear_custom.weight).float()
        bias_tensor = torch.from_numpy(linear_custom.bias).float()
        out_torch = F.linear(in_tensor, weight_tensor, bias_tensor)

        assert out_custom.shape == out_torch.size()
        assert torch.allclose(
            torch.from_numpy(out_custom).float(), out_torch, atol=1e-4
        )
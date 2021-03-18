import numpy as np
import pytest
import torch
import torch.nn.functional as F

import letter_recognition.nn.layers as nn_custom


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("out_channels", [3, 10])
@pytest.mark.parametrize("in_H, in_W", [(28, 28), (21, 5)])
@pytest.mark.parametrize("kernel_size", [3, (1, 4)])
@pytest.mark.parametrize("padding", [0, 2, (2, 1)])
class TestConv2d:
    @pytest.mark.parametrize("bias", [False, True])
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

    @pytest.mark.here
    def test_backward(
        self,
        batch_size,
        in_channels,
        in_H,
        in_W,
        out_channels,
        kernel_size,
        padding,
    ):
        in_shape = (batch_size, in_channels, in_H, in_W)
        in_array = np.random.randint(0, 256, in_shape).astype("float")
        conv2d_custom = nn_custom.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=True
        )
        in_tensor = torch.from_numpy(in_array).float()
        in_tensor.requires_grad_(True)

        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        weight_tensor.requires_grad_(True)
        bias_tensor = torch.from_numpy(conv2d_custom.bias).float()
        bias_tensor.requires_grad_(True)
        out_torch = F.conv2d(
            in_tensor, weight_tensor, bias=bias_tensor, padding=padding
        )

        out_torch.retain_grad()
        final = out_torch * 2
        final.sum().backward()

        input_gradient_torch = in_tensor.grad
        weight_gradient_torch = weight_tensor.grad
        bias_gradient_torch = bias_tensor.grad

        (
            input_gradient_custom,
            weight_gradient_custom,
            bias_gradient_custom,
        ) = conv2d_custom.backward(out_torch.grad.numpy(), in_array)

        assert input_gradient_custom.shape == input_gradient_torch.size()
        assert weight_gradient_custom.shape == weight_gradient_torch.size()
        assert bias_gradient_custom.shape == bias_gradient_torch.size()

        assert torch.allclose(
            torch.from_numpy(input_gradient_custom).float(),
            input_gradient_torch,
            atol=1e-4,
        )
        assert torch.allclose(
            torch.from_numpy(weight_gradient_custom).float(),
            weight_gradient_torch,
            atol=1e-4,
        )
        assert torch.allclose(
            torch.from_numpy(bias_gradient_custom).float(),
            bias_gradient_torch,
            atol=1e-4,
        )


@pytest.mark.parametrize("batch_size", [1, 4, 50])
@pytest.mark.parametrize("in_features", [3, 100, 500])
@pytest.mark.parametrize("out_features", [2, 50, 400])
class TestLinear:
    @pytest.mark.parametrize("bias", [False, True])
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

    def test_backward(self, batch_size, in_features, out_features):
        in_shape = (batch_size, in_features)
        in_array = np.random.randint(0, 256, in_shape).astype("float")
        in_tensor = torch.from_numpy(in_array).float()
        in_tensor.requires_grad_(True)

        linear_custom = nn_custom.Linear(in_features, out_features, bias=True)

        weight_tensor = torch.from_numpy(linear_custom.weight).float()
        weight_tensor.requires_grad_(True)
        bias_tensor = torch.from_numpy(linear_custom.bias).float()
        bias_tensor.requires_grad_(True)
        out_torch = F.linear(in_tensor, weight_tensor, bias_tensor)

        out_torch.retain_grad()
        final = out_torch * 2
        final.sum().backward()

        input_gradient_torch = in_tensor.grad
        weight_gradient_torch = weight_tensor.grad
        bias_gradient_torch = bias_tensor.grad

        (
            input_gradient_custom,
            weight_gradient_custom,
            bias_gradient_custom,
        ) = linear_custom.backward(out_torch.grad.numpy(), in_array)

        assert input_gradient_custom.shape == input_gradient_torch.size()
        assert weight_gradient_custom.shape == weight_gradient_torch.size()
        assert bias_gradient_custom.shape == bias_gradient_torch.size()

        assert torch.allclose(
            torch.from_numpy(input_gradient_custom).float(),
            input_gradient_torch,
            atol=1e-4,
        )
        assert torch.allclose(
            torch.from_numpy(weight_gradient_custom).float(),
            weight_gradient_torch,
            atol=1e-4,
        )
        assert torch.allclose(
            torch.from_numpy(bias_gradient_custom).float(),
            bias_gradient_torch,
            atol=1e-4,
        )

    @pytest.mark.parametrize("bias", [False, True])
    def test_integration(self, batch_size, in_features, out_features, bias):
        # For now, just checking if there are no errors.
        in_shape = (batch_size, in_features)
        in_array = np.random.randint(0, 256, in_shape).astype("float")
        linear = nn_custom.Linear(in_features, out_features, bias)
        out = linear.forward(in_array)
        dout = np.full_like(out, 2)
        dx, dW, db = linear.backward(dout, in_array)
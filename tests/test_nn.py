import numpy as np
import pytest
import torch
import torch.nn
import torch.nn.functional as F

from tests import RNG
import letter_recognition.nn.activation as activation_custom
import letter_recognition.nn.layers as nn_custom
import letter_recognition.nn.loss as loss_custom


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("out_channels", [3, 10])
@pytest.mark.parametrize("in_H, in_W", [(28, 28), (21, 5)])
@pytest.mark.parametrize("kernel_size", [3, (1, 4)])
@pytest.mark.parametrize("padding", [0, 2, (2, 1)])
@pytest.mark.parametrize("bias", [False, True])
class TestConv2d:
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
        in_array = RNG.integers(-127, 128, in_shape).astype("float")
        conv2d_custom = nn_custom.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=bias
        )
        out_custom = conv2d_custom.forward(in_array)

        in_tensor = torch.from_numpy(in_array).float()
        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        bias_tensor = torch.from_numpy(conv2d_custom.bias).float()
        out_torch = F.conv2d(
            in_tensor, weight_tensor, bias=bias_tensor, padding=padding
        )

        assert np.allclose(out_custom, out_torch, atol=1e-4)

    def test_backward(
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
        in_array = RNG.integers(-127, 128, in_shape).astype("float")
        conv2d_custom = nn_custom.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=bias
        )

        in_tensor = torch.from_numpy(in_array).float()
        in_tensor.requires_grad_()
        weight_tensor = torch.from_numpy(conv2d_custom.weight).float()
        weight_tensor.requires_grad_()
        bias_tensor = torch.from_numpy(conv2d_custom.bias).float()
        bias_tensor.requires_grad_()
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

        assert np.allclose(input_gradient_custom, input_gradient_torch, atol=1e-4)
        assert np.allclose(weight_gradient_custom, weight_gradient_torch, atol=1e-4)
        assert np.allclose(bias_gradient_custom, bias_gradient_torch, atol=1e-4)


@pytest.mark.parametrize("batch_size", [1, 4, 64])
@pytest.mark.parametrize("in_features", [3, 100, 500])
@pytest.mark.parametrize("out_features", [2, 50, 400])
class TestLinear:
    @pytest.mark.parametrize("bias", [False, True])
    def test_forward(self, batch_size, in_features, out_features, bias):
        in_shape = (batch_size, in_features)
        in_array = RNG.integers(-127, 128, in_shape).astype("float")
        linear_custom = nn_custom.Linear(in_features, out_features, bias)
        out_custom = linear_custom.forward(in_array)

        in_tensor = torch.from_numpy(in_array).float()
        weight_tensor = torch.from_numpy(linear_custom.weight).float()
        bias_tensor = torch.from_numpy(linear_custom.bias).float()
        out_torch = F.linear(in_tensor, weight_tensor, bias_tensor)

        assert np.allclose(out_custom, out_torch, atol=1e-4)

    def test_backward(self, batch_size, in_features, out_features):
        in_shape = (batch_size, in_features)
        in_array = RNG.integers(-127, 128, in_shape).astype("float")
        linear_custom = nn_custom.Linear(in_features, out_features, bias=True)

        in_tensor = torch.from_numpy(in_array).float()
        in_tensor.requires_grad_()
        weight_tensor = torch.from_numpy(linear_custom.weight).float()
        weight_tensor.requires_grad_()
        bias_tensor = torch.from_numpy(linear_custom.bias).float()
        bias_tensor.requires_grad_()
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

        assert np.allclose(input_gradient_custom, input_gradient_torch, atol=1e-4)
        assert np.allclose(weight_gradient_custom, weight_gradient_torch, atol=1e-4)
        assert np.allclose(bias_gradient_custom, bias_gradient_torch, atol=1e-4)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("in_H, in_W", [(10, 10), (21, 5), (5, 21)])
@pytest.mark.parametrize("kernel_size", [2, (3, 2), (2, 3)])
@pytest.mark.parametrize("padding", [0, 1, (1, 0), (0, 1)])
@pytest.mark.parametrize("ceil_mode", [False, True])
class TestMaxPool2d:
    def test_forward(
        self, batch_size, in_channels, in_H, in_W, kernel_size, padding, ceil_mode
    ):
        in_shape = (batch_size, in_channels, in_H, in_W)
        in_array = RNG.integers(-127, 128, in_shape).astype("float")
        maxpool_custom = nn_custom.MaxPool2d(
            kernel_size, padding=padding, ceil_mode=ceil_mode
        )
        out_custom, idx_custom = maxpool_custom.forward(in_array)

        in_tensor = torch.from_numpy(in_array).float()
        out_torch, idx_torch = F.max_pool2d_with_indices(
            in_tensor, kernel_size, padding=padding, ceil_mode=ceil_mode
        )

        assert np.allclose(out_custom, out_torch, atol=1e-4)
        assert np.allclose(idx_custom, idx_torch, atol=1e-4)

    def test_backward(
        self, batch_size, in_channels, in_H, in_W, kernel_size, padding, ceil_mode
    ):
        in_shape = (batch_size, in_channels, in_H, in_W)
        in_array = RNG.integers(-127, 128, in_shape).astype("float")
        maxpool_custom = nn_custom.MaxPool2d(
            kernel_size, padding=padding, ceil_mode=ceil_mode
        )
        _, idx_custom = maxpool_custom.forward(in_array)

        in_tensor = torch.from_numpy(in_array).float()
        in_tensor.requires_grad_()
        out_torch = F.max_pool2d_with_indices(
            in_tensor, kernel_size, padding=padding, ceil_mode=ceil_mode
        )[0]
        out_torch.retain_grad()
        final = out_torch * 2
        final.sum().backward()
        input_gradient_torch = in_tensor.grad

        input_gradient_custom = maxpool_custom.backward(
            out_torch.grad.numpy(), in_array, idx_custom
        )

        assert np.allclose(input_gradient_custom, input_gradient_torch, atol=1e-4)


@pytest.mark.parametrize("in_shape", [(50, 3, 5, 5), (5, 5), (128, 128, 3)])
class TestReLU:
    def test_forward(self, in_shape):
        in_array = RNG.integers(-127, 128, in_shape).astype("float")
        relu_custom = activation_custom.ReLU()
        out_custom = relu_custom.forward(in_array)

        in_tensor = torch.from_numpy(in_array).float()
        out_torch = F.relu(in_tensor)

        assert np.allclose(out_custom, out_torch, atol=1e-4)

    def test_backward(self, in_shape):
        in_array = RNG.integers(-127, 128, in_shape).astype("float")
        relu_custom = activation_custom.ReLU()

        in_tensor = torch.from_numpy(in_array).float()
        in_tensor.requires_grad_()
        out_torch = F.relu(in_tensor)
        out_torch.retain_grad()
        final = out_torch * 2
        final.sum().backward()
        input_gradient_torch = in_tensor.grad

        input_gradient_custom = relu_custom.backward(out_torch.grad.numpy(), in_array)

        assert np.allclose(input_gradient_custom, input_gradient_torch, atol=1e-4)


@pytest.mark.parametrize("batch_size", [1, 4, 64])
@pytest.mark.parametrize("class_count", [5, 26])
@pytest.mark.parametrize("use_weight", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("max_value", [1, 255])
class TestCrossEntropy:
    def test_calculate(self, batch_size, class_count, use_weight, reduction, max_value):
        weight_array = None
        weight_torch = None
        if use_weight:
            weight_array = RNG.uniform(size=class_count)
            weight_torch = torch.from_numpy(weight_array).float()

        in_shape = (batch_size, class_count)
        predicted_array = RNG.integers(0, max_value + 1, in_shape).astype("float")
        class_array = RNG.integers(0, class_count, batch_size)
        ce_custom = loss_custom.CrossEntropy(weight=weight_array, reduction=reduction)
        out_custom = ce_custom.calculate(predicted_array, class_array)

        predicted_tensor = torch.from_numpy(predicted_array).float()
        class_tensor = torch.from_numpy(class_array)
        ce_torch = torch.nn.CrossEntropyLoss(weight=weight_torch, reduction=reduction)
        out_torch = ce_torch(predicted_tensor, class_tensor)

        assert np.allclose(out_custom, out_torch, atol=1e-4)

    def test_backward(self, batch_size, class_count, use_weight, reduction, max_value):
        weight_array = None
        weight_torch = None
        if use_weight:
            weight_array = RNG.uniform(size=class_count)
            weight_torch = torch.from_numpy(weight_array).float()

        in_shape = (batch_size, class_count)
        predicted_array = RNG.integers(0, max_value + 1, in_shape).astype("float")
        class_array = RNG.integers(0, class_count, batch_size)
        ce_custom = loss_custom.CrossEntropy(weight=weight_array, reduction=reduction)
        grad_custom = ce_custom.backward(predicted_array, class_array)

        predicted_tensor = torch.from_numpy(predicted_array).float()
        predicted_tensor.requires_grad_()
        class_tensor = torch.from_numpy(class_array)
        ce_torch = torch.nn.CrossEntropyLoss(weight=weight_torch, reduction=reduction)
        out_torch = ce_torch(predicted_tensor, class_tensor)
        out_torch.sum().backward()
        grad_torch = predicted_tensor.grad

        assert np.allclose(grad_custom, grad_torch, atol=1e-4)


@pytest.mark.parametrize("batch_size", [1, 4, 64])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("max_value", [1, 255])
class TestMAE:
    def test_calculate(self, batch_size, reduction, max_value):
        predicted_array = RNG.integers(0, max_value + 1, batch_size).astype("float")
        target_array = RNG.integers(0, max_value + 1, batch_size).astype("float")
        mae_custom = loss_custom.MAE(reduction)
        out_custom = mae_custom.calculate(predicted_array, target_array)

        predicted_tensor = torch.from_numpy(predicted_array).float()
        target_tensor = torch.from_numpy(target_array).float()
        mae_torch = torch.nn.L1Loss(reduction=reduction)
        out_torch = mae_torch(predicted_tensor, target_tensor)

        assert out_custom.shape == out_torch.size()
        assert np.allclose(out_custom, out_torch, atol=1e-4)

    def test_backward(self, batch_size, reduction, max_value):
        predicted_array = RNG.integers(0, max_value + 1, batch_size).astype("float")
        target_array = RNG.integers(0, max_value + 1, batch_size).astype("float")
        mae_custom = loss_custom.MAE(reduction)
        grad_custom = mae_custom.backward(predicted_array, target_array)

        predicted_tensor = torch.from_numpy(predicted_array).float()
        predicted_tensor.requires_grad_()
        target_tensor = torch.from_numpy(target_array).float()
        mae_torch = torch.nn.L1Loss(reduction=reduction)
        out_torch = mae_torch(predicted_tensor, target_tensor)
        out_torch.sum().backward()
        grad_torch = predicted_tensor.grad

        assert np.allclose(grad_custom, grad_torch, atol=1e-4)


@pytest.mark.parametrize("batch_size", [1, 4, 64])
@pytest.mark.parametrize("class_count", [5, 26])
@pytest.mark.parametrize("max_value", [1, 255])
class TestSoftmax:
    def test_forward(self, batch_size, class_count, max_value):
        in_shape = (batch_size, class_count)
        in_array = RNG.integers(0, max_value + 1, in_shape).astype("float")
        softmax_custom = activation_custom.Softmax()
        out_custom = softmax_custom.forward(in_array)

        in_tensor = torch.from_numpy(in_array).float()
        out_torch = F.softmax(in_tensor, dim=1)

        assert np.allclose(out_custom, out_torch, atol=1e-4)

import string

import numpy as np
import matplotlib.pyplot as plt

import letter_recognition.nn.layers as nn
import letter_recognition.nn.activation as activation
from letter_recognition import RNG

# Predict
if __name__ == "__main__":
    # Model parameters
    conv1_out_channels = 6
    conv1_kernel_size = (5, 5)
    maxpool1_size = (2, 2)
    conv2_out_channels = 16
    conv2_kernel_size = (5, 5)
    maxpool2_size = (2, 2)
    linear1_in_feat = conv2_out_channels * 4 * 4
    linear1_out_feat = 120
    linear2_out_feat = 84

    # Model architecture
    conv1 = nn.Conv2d(1, conv1_out_channels, conv1_kernel_size)
    maxpool1 = nn.MaxPool2d(maxpool1_size)
    conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, conv2_kernel_size)
    maxpool2 = nn.MaxPool2d(maxpool2_size)
    linear1 = nn.Linear(linear1_in_feat, linear1_out_feat)
    linear2 = nn.Linear(linear1_out_feat, linear2_out_feat)
    linear3 = nn.Linear(linear2_out_feat, 26)
    relu = activation.ReLU()
    softmax = activation.Softmax()

    # Model loading
    with open("letter_recognition/data/models/lenet5.npz", "rb") as f:
        model_params = np.load(f)
        conv1.weight = model_params["conv1weight"]
        conv1.bias = model_params["conv1bias"]
        conv2.weight = model_params["conv2weight"]
        conv2.bias = model_params["conv2bias"]
        linear1.weight = model_params["linear1weight"]
        linear1.bias = model_params["linear1bias"]
        linear2.weight = model_params["linear2weight"]
        linear2.bias = model_params["linear2bias"]
        linear3.weight = model_params["linear3weight"]
        linear3.bias = model_params["linear3bias"]

    # Data loading
    with open("letter_recognition/data/dataset/data.npz", "rb") as f:
        data = np.load(f)
        images = data["X"]
        # labels = data["Y"]

    # Choose random image
    i = RNG.integers(0, len(images))
    x = images[i : i + 1]

    # Go through the model
    out_conv1 = conv1.forward(x)
    out_conv1_relu = relu.forward(out_conv1)
    out_maxpool1, _ = maxpool1.forward(out_conv1_relu)
    out_conv2 = conv2.forward(out_maxpool1)
    out_conv2_relu = relu.forward(out_conv2)
    out_maxpool2, _ = maxpool2.forward(out_conv2_relu)
    out_maxpool2_reshaped = out_maxpool2.reshape(out_maxpool2.shape[0], linear1_in_feat)
    out_linear1 = linear1.forward(out_maxpool2_reshaped)
    out_linear1_relu = relu.forward(out_linear1)
    out_linear2 = linear2.forward(out_linear1_relu)
    out_linear2_relu = relu.forward(out_linear2)
    out_linear3 = linear3.forward(out_linear2_relu)

    # Map output (probabilities) to letters
    letters = string.ascii_uppercase
    probabilities = {
        letters[i]: round(softmax.forward(out_linear3)[0, i], 2)
        for i in range(len(letters))
    }
    print("Predicted letter:", max(probabilities, key=probabilities.get))
    print(probabilities)

    fig, ax = plt.subplots()
    ax.imshow(x[0, 0], cmap="gray")
    plt.show()
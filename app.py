import base64
import io
import string
from typing import Tuple, Union

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps

import letter_recognition.nn.layers as nn
import letter_recognition.nn.activation as activation

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

# Model hyperparameters
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
conv1.weight = np.load("letter_recognition/data/models/lenet5/conv1weight.npy")
conv1.bias = np.load("letter_recognition/data/models/lenet5/conv1bias.npy")
conv2.weight = np.load("letter_recognition/data/models/lenet5/conv2weight.npy")
conv2.bias = np.load("letter_recognition/data/models/lenet5/conv2bias.npy")
linear1.weight = np.load("letter_recognition/data/models/lenet5/linear1weight.npy")
linear1.bias = np.load("letter_recognition/data/models/lenet5/linear1bias.npy")
linear2.weight = np.load("letter_recognition/data/models/lenet5/linear2weight.npy")
linear2.bias = np.load("letter_recognition/data/models/lenet5/linear2bias.npy")
linear3.weight = np.load("letter_recognition/data/models/lenet5/linear3weight.npy")
linear3.bias = np.load("letter_recognition/data/models/lenet5/linear3bias.npy")


def transform_image(image_b64: Union[str, bytes]) -> np.ndarray:
    """Transforms an image to the input expected by the model.

    Parameters
    ----------
    image_encoded : string or bytes
        Base-64 string or bytes representation of the image.

    Returns
    -------
    np.ndarray
        Transformed image.
    """
    # Decoding
    if isinstance(image_b64, str):
        image_b64 = image_b64.encode()
    image_bytes = base64.b64decode(image_b64)
    image_pil = Image.open(io.BytesIO(image_bytes))

    # Processing
    if image_pil.size != (28, 28):
        image_pil = image_pil.resize((28, 28))
    image_pil = ImageOps.grayscale(image_pil)
    image = np.asarray(image_pil)
    threshold = ((np.max(image) + np.mean(image)) / 2) * (
        1 - 0.2 * (1 - np.std(image) / 128)
    )
    image_binarized = np.where(image > threshold, 1.0, 0.0)

    return image_binarized.reshape(1, 1, 28, 28)


def get_prediction(image_b64: Union[str, bytes]) -> Tuple[str, np.ndarray]:
    """Gets the prediction for a given numpy image.

    Parameters
    ----------
    image_encoded : string or bytes
        Base-64 string or bytes representation of the image.

    Returns
    -------
    tuple of str and numpy array
        The predicted letter and the probabilities for each letter.
    """
    # Data loading
    x = transform_image(image_b64)

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
    return (max(probabilities, key=probabilities.get), probabilities)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        in_json = request.get_json(force=True)
        image_b64 = in_json["image"]
        predicted, probabilities = get_prediction(image_b64)
        response = jsonify({"predicted": predicted, "probabilities": probabilities})
        return response
    if request.method == "GET":
        return "Letter Recognition Model API"

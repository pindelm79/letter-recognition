import io
import string
from typing import Tuple

from flask import Flask, jsonify, request
import numpy as np
from PIL import Image, ImageOps

import letter_recognition.nn.layers as nn
import letter_recognition.nn.activation as activation

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, World 3"
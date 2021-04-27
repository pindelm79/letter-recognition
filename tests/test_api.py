import io
import base64
import string

import numpy as np
from PIL import Image, ImageOps
import requests

from tests import RNG


def test_model():
    # Load example image
    with open("letter_recognition/data/dataset/data.npz", "rb") as f:
        data = np.load(f)
        images = data["X"]
        labels = data["Y"]
    i = RNG.integers(0, len(images))
    image = images[i, 0] * 255

    # Save the image to png
    image_pil = Image.fromarray(image)
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    image_pil.save("letter_recognition/data/temp/tmp.png")

    # Encode image as b64
    with open("letter_recognition/data/temp/tmp.png", "rb") as f:
        img_encoded = base64.b64encode(f.read())

    # Decode image from b64
    img_decoded = base64.b64decode(img_encoded)

    response = requests.post(
        "http://127.0.0.1:5000/",
        files={"file": img_decoded},
    )

    letter_mapping = dict(zip(range(26), string.ascii_uppercase))
    print(response.json())
    assert letter_mapping[labels[i]] == response.json()["predicted"]

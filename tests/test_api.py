import string

import numpy as np
from PIL import Image
import requests

from tests import RNG


def test_model():
    with open("letter_recognition/data/dataset/data.npz", "rb") as f:
        data = np.load(f)
        images = data["X"]
        labels = data["Y"]
    i = RNG.integers(0, len(images))
    image = images[i, 0] * 255

    image_pil = Image.fromarray(image)
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    image_pil.save("letter_recognition/data/temp/tmp.png")

    response = requests.post(
        "https://letterrecognitionapi.azurewebsites.net/",
        files={"file": open("letter_recognition/data/temp/tmp.png", "rb")},
    )

    letter_mapping = dict(zip(range(26), string.ascii_uppercase))
    print(response.json())
    assert letter_mapping[labels[i]] == response.json()["predicted"]

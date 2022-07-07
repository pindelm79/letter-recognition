import os

import numpy as np


_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data_path(path):
    return os.path.join(_ROOT, "data", path)


def load_np_file(relative_data_path):
    absolute_path = get_data_path(relative_data_path)
    with open(absolute_path, "rb") as f:
        content = np.load(f)
    return content

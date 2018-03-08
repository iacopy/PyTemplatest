"""
Utility functions for the puzzle module: loading and saving images.
"""

# 3rd party
import numpy as np
from PIL import Image


def load_image_as_grayscale(src):
    """
    Open and convert image to 2D grayscale array.
    """
    image = Image.open(src).convert('L')
    ary = np.asarray(image)
    return ary


def load_image(src):
    """
    Open and convert image to array.
    """
    image = Image.open(src)
    ary = np.asarray(image)
    return ary


def save_image(ary, dst):
    """
    Save an `ary` numpy array image to disk in `dst` file path.
    """
    Image.fromarray(ary.astype(np.uint8)).save(dst)

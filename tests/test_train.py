import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import tensorflow as tf

from app.train import create_model, get_data


def test_create_model():
    """
    Tests that the create_model function returns a Sequential model object.
    """
    model = create_model()
    assert isinstance(model, tf.keras.Sequential)


def test_get_data():
    """
    Tests that the get_data function returns the expected input and output arrays.
    """
    xs, ys = get_data()
    assert len(xs) == len(ys)
    assert isinstance(xs, np.ndarray)
    assert isinstance(ys, np.ndarray)

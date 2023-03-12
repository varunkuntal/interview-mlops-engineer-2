import os

# Turning warning messages off for tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

import numpy as np
import tensorflow as tf

# Adding app module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.predict import load_saved_model, predict

MODEL_DIR = 'app/model'
MODEL_FILENAME = 'my_best_model.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Define the test data and expected results
test_data = [-5.0, 0.0, 5.0]
expected_results = np.array([[-0.08138456], [0.12002773], [0.32144]])


def test_load_saved_model():
    """
    Tests load_saved_model function loads a saved model from the specified file path.
    """
    model = load_saved_model(MODEL_PATH)
    assert isinstance(model, tf.keras.Sequential)


def test_predict():
    """
    Tests that the predict function produces expected predictions on test data.
    """
    model = load_saved_model(MODEL_PATH)
    predictions = predict(model, test_data)
    assert np.allclose(predictions, expected_results)

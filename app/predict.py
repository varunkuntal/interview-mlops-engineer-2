import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_DIR = 'model'
MODEL_FILENAME = 'my_best_model.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def load_saved_model(file_path: str = 'my_best_model.h5') -> 'tf.keras.Sequential':
    """
    Loads a saved model from the specified file path and returns it.
    """
    loaded_model = tf.keras.models.load_model(file_path)
    logger.info(f'Model loaded from {file_path}!')
    return loaded_model


def predict(saved_model: 'tf.keras.Sequential', input_value: list[float]) -> np.ndarray:
    """
    Make predictions on the given inputs and returns the predicted outputs as a numpy array.
    """
    predicted_output = saved_model.predict(input_value)
    return predicted_output


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    tf_model = load_saved_model(MODEL_PATH)
    prediction = predict(tf_model, [10])
    logger.info(f'Prediction: {prediction}')

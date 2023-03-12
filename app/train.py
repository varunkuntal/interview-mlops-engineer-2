import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging

import numpy as np
import tensorflow as tf

MODEL_DIR = './model'
MODEL_FILENAME = 'my_best_model.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_model() -> tf.keras.Sequential:
    """
    Creates and returns a simple neural network model with one dense layer and an SGD optimizer.
    """
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model


def get_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Provides the data for training the neural network. Returns a tuple containing two numpy arrays:
    - xs: An array of input values
    - ys: An array of output values
    """
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
    return xs, ys


def train_model(
    model: tf.keras.Sequential, xs: np.ndarray, ys: np.ndarray, epochs: int = 500
) -> None:
    """
    Trains the neural network model on the provided data for the specified number of epochs (default: 500).
    """
    model.fit(xs, ys, epochs)
    logger.info('Model trained successfully!')


def save_model_custom(
    model: tf.keras.Sequential, filename: str = 'my_best_model.h5'
) -> None:
    """
    Saves the provided model to the specified file (default: 'my_best_model.h5').
    """
    tf.keras.models.save_model(model, filename)
    logger.info(f'Model saved to file: {filename}')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    os.makedirs(MODEL_DIR, exist_ok=True)
    tf_model = create_model()
    xs, ys = get_data()
    train_model(tf_model, xs, ys)
    save_model_custom(tf_model, MODEL_PATH)

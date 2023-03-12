import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging

import tensorflow as tf
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from app.predict import load_saved_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_DIR = 'model'
MODEL_FILENAME = 'my_best_model.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

app = FastAPI(title="MLOps Interview REST API",
    version="0.0.1",)

# Load the model at startup and store it in a global variable
tf_model = load_saved_model(MODEL_PATH)


@app.get('/prediction/{input_value}')
async def prediction(input_value: float):
    """
    Make predictions on the given input and returns the predicted output as a JSON response.
    """
    predicted_output = tf_model.predict([[input_value]])
    logger.info(f'API called, prediction: {predicted_output}')
    return jsonable_encoder({'prediction': predicted_output.tolist()})

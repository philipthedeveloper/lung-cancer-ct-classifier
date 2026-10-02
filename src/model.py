import json
from pathlib import Path

import tensorflow as tf
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

# Fallback mapping if models/class_names.json is missing (alphabetical folder order)
CLASS_NAMES = [
    'Adenocarcinoma',
    'Large Cell Carcinoma',
    'Normal',
    'Squamous Cell Carcinoma',
]


def load_model(model_path=MODELS_DIR / 'model.keras'):
    """Load the trained model produced by train.py."""
    global CLASS_NAMES
    names_path = MODELS_DIR / 'class_names.json'
    if names_path.exists():
        CLASS_NAMES = json.loads(names_path.read_text())
    return tf.keras.models.load_model(model_path)


def classify_image(model, processed_image):
    """Classify an image using the loaded model. Returns (class_id, confidence)."""
    predictions = model.predict(processed_image, verbose=0)[0]
    class_id = int(np.argmax(predictions))
    return class_id, float(predictions[class_id])


def get_class_name(class_id):
    """Get the class name corresponding to the class ID."""
    return CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else 'Unknown'

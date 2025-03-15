import tensorflow as tf
import numpy as np

# Dictionary mapping class numbers to names
CLASS_NAMES = {
    0: 'Adenocarcinoma',
    1: 'Large Cell Carcinoma',
    2: 'Normal',
    3: 'Squamous Cell Carcino'
}


def load_model(model_path='../models/model.h5'):
    """Load the pre-trained CNN model."""
    model = tf.keras.models.load_model(model_path)
    return model


def classify_image(model, processed_image):
    """Classify an image using the loaded model."""
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]  # Return the first (and only) class


def get_class_name(class_id):
    """Get the class name corresponding to the class ID."""
    return CLASS_NAMES.get(class_id, 'Unknown')

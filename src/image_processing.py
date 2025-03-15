import cv2
import numpy as np


def preprocess_image(image_path):
    """Preprocess the image to match the input format expected by the model."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to the input shape expected by the model
    preprocessed_path = "preprocessed_image.jpg"
    cv2.imwrite(preprocessed_path, image)  # Save the preprocessed image
    image = image / 255.0  # Rescale the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image, preprocessed_path

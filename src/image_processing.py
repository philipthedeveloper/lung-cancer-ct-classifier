import cv2
import numpy as np


def preprocess_image(image_path):
    """Preprocess the image to match the input format expected by the model."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to the input shape expected by the model
    preprocessed_path = "preprocessed_image.jpg"
    cv2.imwrite(preprocessed_path, image)  # Save the preprocessed image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR; the model was trained on RGB
    # No /255 rescaling: EfficientNet normalises raw 0-255 pixels internally
    image = np.expand_dims(image.astype('float32'), axis=0)  # Add batch dimension
    return image, preprocessed_path

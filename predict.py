from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved model
model = load_model('water_pollution_model.h5')

# Function to load and preprocess an image
def load_and_process_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Rescale to [0, 1]

# Example usage: Predict on a single image
def predict_single_image(image_path):
    processed_image = load_and_process_image(image_path)
    prediction = model.predict(processed_image)
    if prediction[0] > 0.5:
        print(f"Predicted: Polluted (confidence: {prediction[0][0]})")
    else:
        print(f"Predicted: Clean (confidence: {1 - prediction[0][0]})")

# Example: Replace 'image_path' with the path to your test image
image_path = r"C:\Users\drisa\Downloads\Screenshot 2024-07-07 183324.png"
predict_single_image(image_path)

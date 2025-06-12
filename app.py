import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the model
try:
    model = tf.keras.models.load_model('models\pneumonia_augmented_model.h5')
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print("[ERROR] Failed to load model:", e)

# Define image preprocessing
def preprocess(image: Image.Image):
    try:
        image = image.convert("L")  # Convert to grayscale
        img = np.array(image)
        img = cv2.resize(img, (150, 150))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)  # Shape: (150, 150, 1)
        img = np.expand_dims(img, axis=0)   # Shape: (1, 150, 150, 1)
        return img
    except Exception as e:
        print("[ERROR] Preprocessing failed:", e)
        return None

# Define prediction function
def predict_pneumonia(image):
    img = preprocess(image)
    if img is None:
        return "Error", "Preprocessing failed"

    try:
        prediction = model.predict(img)
        prob = prediction[0][0]
        if prob > 0.5:
            return "Pneumonia", f"{prob * 100:.2f}% confidence"
        else:
            return "Normal", f"{(1 - prob) * 100:.2f}% confidence"
    except Exception as e:
        print("[ERROR] Prediction failed:", e)
        return "Error", str(e)

# Gradio interface
iface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(label="Prediction"), gr.Textbox(label="Confidence")],
    title="Pneumonia Detection from Chest X-Ray",
    description="Upload a chest X-ray image. The model will predict whether it shows pneumonia or is normal.",
)

if __name__ == "__main__":
    iface.launch(debug=True)  # debug=True shows more logs in terminal

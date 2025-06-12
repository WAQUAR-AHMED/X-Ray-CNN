# Pneumonia Detection from Chest X-Ray Images

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images into **Pneumonia** or **Normal**.

## How to Run

1. Clone the repo.
2. Install dependencies:
3. pip install -r requirements.txt
4. Run the app:

# 🩺 Pneumonia Detection from Chest X-ray using Deep Learning

This project uses a Convolutional Neural Network (CNN) to detect **Pneumonia** from chest X-ray images. The model was trained on the widely-used Kaggle `chest_xray` dataset and deployed as a web application using Gradio and Hugging Face Spaces.

👉 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/waquarahmed/X-Ray)**  
*Upload a chest X-ray and get instant diagnosis (Normal / Pneumonia).*

---

## 🚀 Features

- 📊 **CNN Model**: Custom-built using TensorFlow/Keras
- 🖼️ **Image Preprocessing**: Normalization, resizing, grayscale conversion
- 🧪 **Data Augmentation**: Boosted accuracy from ~73% to ~88%
- 🌐 **Live Deployment**: Web interface with Gradio
- ☁️ **Free Hosting**: Powered by Hugging Face Spaces

---

## 📁 Dataset

- **Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Structure**:
chest_xray/
├── train/
├── val/
└── test/


Each folder contains X-ray images classified into:
- `NORMAL`
- `PNEUMONIA`

---

## 🧠 Model Architecture

- Input shape: `(150, 150, 1)` – grayscale resized images
- Layers:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dense layers with Dropout
- Final layer: Sigmoid (binary classification)

- Loss Function: `binary_crossentropy`
- Optimizer: `Adam`

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy / OpenCV / Matplotlib**
- **Gradio** – web interface
- **Hugging Face Spaces** – deployment

---

## 🖥️ How to Use (Locally)

1. Clone the repo:
 ```bash
 git clone https://github.com/WAQUAR-AHMED/X-Ray-CNN.git
 cd pneumonia-xray-detector

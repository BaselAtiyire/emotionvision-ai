import os
import urllib.request
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("EmotionVision — Facial Emotion Recognition")

CLASS_NAMES = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

MODEL_PATH = "models/cnn_emotion_model.keras"

MODEL_URL = "https://drive.google.com/uc?id=1NvpU_AfmHOLgiZm6N0MX3MztdNA0COti"

os.makedirs("models", exist_ok=True)

# download model if missing
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.write("Model downloaded")

# load model
model = tf.keras.models.load_model(MODEL_PATH)

uploaded = st.file_uploader("Upload a face image", type=["png","jpg","jpeg"])

if uploaded is not None:

    img = Image.open(uploaded).convert("L").resize((48,48))

    st.image(img, caption="Processed Image")

    img = np.array(img)/255.0
    img = img.reshape(1,48,48,1)

    preds = model.predict(img)[0]
    pred = np.argmax(preds)

    st.subheader(f"Prediction: {CLASS_NAMES[pred]}")

    st.bar_chart({CLASS_NAMES[i]:float(preds[i]) for i in range(len(CLASS_NAMES))})
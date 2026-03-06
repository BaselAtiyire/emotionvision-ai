import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

st.title("EmotionVision — Facial Emotion Recognition")

model = tf.keras.models.load_model("models/cnn_emotion_model.keras")

uploaded = st.file_uploader("Upload a face image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L").resize((48, 48))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))

    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    st.image(img, caption="48x48 Grayscale", width=200)
    st.write(f"**Prediction:** {CLASS_NAMES[pred_idx]}")
    st.write("Probabilities:")
    for i, p in enumerate(probs):
        st.write(f"- {CLASS_NAMES[i]}: {p:.3f}")
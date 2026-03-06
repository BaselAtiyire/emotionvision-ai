import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="EmotionVision",
    page_icon="📷",
    layout="wide",
)

st.title("EmotionVision — Facial Emotion Recognition")
st.caption("Upload an image or use your browser camera for emotion prediction.")

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "cnn_emotion_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=1NvpU_AfmHOLgiZm6N0MX3MztdNA0COti"


def ensure_model_available() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


@st.cache_resource
def load_model() -> tf.keras.Model:
    ensure_model_available()
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(img: Image.Image):
    processed = img.convert("L").resize((48, 48))
    arr = np.array(processed).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    return processed, arr


def predict_emotion(model: tf.keras.Model, arr: np.ndarray):
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def confidence_label(score: float) -> str:
    if score >= 0.80:
        return "Very High"
    if score >= 0.60:
        return "High"
    if score >= 0.40:
        return "Moderate"
    return "Low"


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    _ = model(img_array, training=False)

    conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.outputs[0]],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy())


def create_gradcam_overlay(processed_img: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    gray_img = np.array(processed_img)
    gray_img_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    heatmap_resized = cv2.resize(heatmap, (48, 48))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(gray_img_bgr, 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def show_prediction_dashboard(model: tf.keras.Model, img: Image.Image, source_label: str) -> None:
    processed_img, arr = preprocess_image(img)
    pred_idx, probs = predict_emotion(model, arr)
    confidence = float(probs[pred_idx])

    top_left, top_mid, top_right = st.columns([1.2, 1, 1])

    with top_left:
        st.subheader(f"{source_label} Image")
        st.image(img, use_container_width=True)

    with top_mid:
        st.subheader("Processed Input")
        st.image(processed_img, caption="48×48 grayscale", width=220)
        st.metric("Predicted Emotion", CLASS_NAMES[pred_idx])
        st.metric("Confidence", f"{confidence:.2%}")
        st.metric("Confidence Level", confidence_label(confidence))

    with top_right:
        st.subheader("Confidence Gauge")
        st.progress(confidence)

        sorted_probs = sorted(
            [(CLASS_NAMES[i], float(probs[i])) for i in range(len(CLASS_NAMES))],
            key=lambda x: x[1],
            reverse=True,
        )

        st.write(f"Top prediction: **{CLASS_NAMES[pred_idx]}**")
        st.write("Top 3 classes:")
        for label, score in sorted_probs[:3]:
            st.write(f"- {label}: {score:.2%}")

    st.divider()

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.subheader("Emotion Probabilities")
        chart_data = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(chart_data)

    with bottom_right:
        st.subheader("Grad-CAM Visualization")
        try:
            last_conv = find_last_conv_layer(model)
            heatmap, _ = make_gradcam_heatmap(arr, model, last_conv)
            overlay = create_gradcam_overlay(processed_img, heatmap)
            st.image(overlay, caption="Grad-CAM heatmap overlay", use_container_width=True)
        except Exception as e:
            st.warning("Grad-CAM could not be generated.")
            st.code(str(e))

    with st.expander("Raw probabilities"):
        for i, p in enumerate(probs):
            st.write(f"{CLASS_NAMES[i]}: {p:.4f}")


try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error("Failed to load the model.")
    st.code(str(e))
    st.stop()

tab1, tab2, tab3 = st.tabs(["Upload Image", "Use Camera", "About"])

with tab1:
    st.subheader("Upload a Face Image")
    uploaded = st.file_uploader(
        "Choose a PNG, JPG, or JPEG image",
        type=["png", "jpg", "jpeg"],
        key="file_upload",
    )

    if uploaded is not None:
        try:
            img = Image.open(uploaded)
            show_prediction_dashboard(model, img, "Uploaded")
        except Exception as e:
            st.error("Could not process uploaded image.")
            st.code(str(e))
    else:
        st.info("Upload a JPG or PNG face image to begin.")

with tab2:
    st.subheader("Use Your Camera")
    st.info(
        "This browser version captures a photo from your camera. "
        "For continuous real-time webcam detection, use the local OpenCV webcam script."
    )

    camera_photo = st.camera_input("Take a picture", key="camera_capture")

    if camera_photo is not None:
        try:
            img = Image.open(camera_photo)
            show_prediction_dashboard(model, img, "Camera")
        except Exception as e:
            st.error("Could not process camera image.")
            st.code(str(e))
    else:
        st.info("Allow camera access in your browser, then take a picture.")

with tab3:
    st.subheader("About This Project")
    st.write(
        """
        **EmotionVision** is a facial emotion recognition app built with:
        - TensorFlow / Keras
        - CNN-based emotion classification
        - Grad-CAM explainability
        - Streamlit dashboard interface

        **Supported emotions:**
        Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

        **Note:**
        The hosted Streamlit version supports browser camera capture.
        For full continuous real-time webcam inference, use your local OpenCV app.
        """
    )
import os
import cv2
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_PATH = "models/cnn_emotion_model.keras"


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Keras 3 fix:
    - Ensure model has been called (has defined outputs)
    - Use model.outputs[0] instead of model.output
    """
    # Ensure the model is "built" by calling it once
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

    conv_outputs = conv_outputs[0]  # (H,W,C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy())


def main():
    os.makedirs("outputs", exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)

    # Use first test image (you can change idx)
    X_test = np.load("data/test_images.npy")
    idx = 0
    img = X_test[idx]  # (48,48)

    # Prepare input
    inp = (img.astype("float32") / 255.0)[None, ..., None]  # (1,48,48,1)

    # Find last conv layer and compute Grad-CAM
    last_conv = find_last_conv_layer(model)
    print("✅ Using last conv layer:", last_conv)

    # Get prediction
    probs = model.predict(inp, verbose=0)[0]
    pred = int(np.argmax(probs))

    heatmap, _ = make_gradcam_heatmap(inp, model, last_conv)

    # Overlay heatmap on original image
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    heatmap_resized = cv2.resize(heatmap, (48, 48))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

    out_path = "outputs/gradcam_overlay.png"
    cv2.imwrite(out_path, overlay)

    print(f"✅ Prediction: {CLASS_NAMES[pred]} ({probs[pred]:.3f})")
    print(f"🖼 Saved: {out_path}")


if __name__ == "__main__":
    main()
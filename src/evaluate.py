import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from src.data_utils import load_data, preprocess_for_cnn

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def main():
    _, _, X_test, y_test = load_data()
    X_test = preprocess_for_cnn(X_test)

    model = tf.keras.models.load_model("models/cnn_emotion_model.keras")
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    print("\n📌 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("outputs", exist_ok=True)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (CNN)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_cnn.png", dpi=200)
    print("\n🖼 Saved: outputs/confusion_matrix_cnn.png")

if __name__ == "__main__":
    main()
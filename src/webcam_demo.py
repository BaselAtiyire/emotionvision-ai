import cv2
import numpy as np
import tensorflow as tf

print("Starting webcam demo...")

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_PATH = "models/cnn_emotion_model.keras"

def preprocess_face_for_cnn(face_gray_48):
    x = face_gray_48.astype("float32") / 255.0
    x = np.expand_dims(x, axis=(0, -1))
    return x

def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    print("Opening webcam...")

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam opened. Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face48 = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)

            inp = preprocess_face_for_cnn(face48)
            probs = model.predict(inp, verbose=0)[0]
            pred = int(np.argmax(probs))
            label = f"{CLASS_NAMES[pred]} ({probs[pred]:.2f})"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        cv2.imshow("EmotionVision Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
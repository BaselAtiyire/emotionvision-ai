import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

from src.data_utils import load_data, preprocess_for_cnn


CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(32, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(),

            layers.Conv2D(256, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.35),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history_and_plots(history, out_dir="outputs", prefix="cnn"):
    os.makedirs(out_dir, exist_ok=True)

    # Save history JSON
    history_path = os.path.join(out_dir, f"{prefix}_train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f)

    # Accuracy plot
    if "accuracy" in history.history and "val_accuracy" in history.history:
        plt.figure()
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        acc_path = os.path.join(out_dir, f"{prefix}_accuracy_curve.png")
        plt.savefig(acc_path, dpi=200)
        plt.close()

    # Loss plot
    if "loss" in history.history and "val_loss" in history.history:
        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        loss_path = os.path.join(out_dir, f"{prefix}_loss_curve.png")
        plt.savefig(loss_path, dpi=200)
        plt.close()

    print(f"✅ Saved history: {history_path}")
    print(f"✅ Saved plots: {prefix}_accuracy_curve.png, {prefix}_loss_curve.png (in {out_dir}/)")


def main():
    # Data
    X_train, y_train, X_test, y_test = load_data()
    X_train = preprocess_for_cnn(X_train)
    X_test = preprocess_for_cnn(X_test)

    # Augmentation
    aug = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="augmentation",
    )

    model = build_model()

    os.makedirs("models", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/cnn_emotion_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]

    history = model.fit(
        aug(X_train),
        y_train,
        validation_split=0.1,
        epochs=35,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Save plots + history
    save_history_and_plots(history, out_dir="outputs", prefix="cnn")

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
    print("💾 Best model saved at: models/cnn_emotion_model.keras")


if __name__ == "__main__":
    main()
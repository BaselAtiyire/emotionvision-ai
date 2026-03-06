import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers


CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def load_data():
    X_train = np.load("data/train_images.npy")
    y_train = np.load("data/train_labels.npy")
    X_test = np.load("data/test_images.npy")
    y_test = np.load("data/test_labels.npy")

    # labels 1..7 -> 0..6 if needed
    if y_train.min() == 1:
        y_train -= 1
    if y_test.min() == 1:
        y_test -= 1
    return X_train, y_train, X_test, y_test


def to_rgb_224(x):
    # (N,48,48) uint8 -> (N,224,224,3) float32
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, -1)       # (N,48,48,1)
    x = np.repeat(x, 3, axis=-1)    # (N,48,48,3)
    x = tf.image.resize(x, (224, 224)).numpy()
    return x


def save_history_and_plots(history, out_dir="outputs", prefix="tl_stage"):
    os.makedirs(out_dir, exist_ok=True)

    history_path = os.path.join(out_dir, f"{prefix}_train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f)

    # Accuracy
    if "accuracy" in history.history and "val_accuracy" in history.history:
        plt.figure()
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_accuracy_curve.png"), dpi=200)
        plt.close()

    # Loss
    if "loss" in history.history and "val_loss" in history.history:
        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_loss_curve.png"), dpi=200)
        plt.close()

    print(f"✅ Saved history: {history_path}")
    print(f"✅ Saved plots: {prefix}_accuracy_curve.png, {prefix}_loss_curve.png (in {out_dir}/)")


def main():
    tf.keras.utils.set_random_seed(42)

    X_train, y_train, X_test, y_test = load_data()
    X_train = to_rgb_224(X_train)
    X_test = to_rgb_224(X_test)

    aug = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base.trainable = False  # Stage 1: freeze backbone

    inputs = layers.Input(shape=(224, 224, 3))
    x = aug(inputs)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(7, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/efficientnetv2_emotion.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]

    # -----------------------
    # Stage 1 training
    # -----------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history1 = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=25,
        batch_size=64,
        callbacks=cbs,
        verbose=1,
    )
    save_history_and_plots(history1, out_dir="outputs", prefix="tl_stage1")

    # -----------------------
    # Stage 2 fine-tuning
    # -----------------------
    base.trainable = True
    # Freeze most layers, fine-tune last ~30
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=12,
        batch_size=64,
        callbacks=cbs,
        verbose=1,
    )
    save_history_and_plots(history2, out_dir="outputs", prefix="tl_stage2")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ EfficientNetV2 Test Accuracy: {acc:.4f}")
    print("💾 Saved: models/efficientnetv2_emotion.keras")


if __name__ == "__main__":
    main()
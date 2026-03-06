import numpy as np

def load_data(
    train_images_path="data/train_images.npy",
    train_labels_path="data/train_labels.npy",
    test_images_path="data/test_images.npy",
    test_labels_path="data/test_labels.npy",
):
    X_train = np.load(train_images_path)
    y_train = np.load(train_labels_path)
    X_test = np.load(test_images_path)
    y_test = np.load(test_labels_path)

    # Convert labels 1..7 -> 0..6 if needed
    if y_train.min() == 1:
        y_train = y_train - 1
    if y_test.min() == 1:
        y_test = y_test - 1

    return X_train, y_train, X_test, y_test

def preprocess_for_cnn(X):
    X = X.astype("float32") / 255.0
    return np.expand_dims(X, axis=-1)  # (N,48,48) -> (N,48,48,1)
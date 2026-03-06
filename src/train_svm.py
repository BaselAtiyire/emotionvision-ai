import os
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from src.data_utils import load_data, preprocess_for_ml
import joblib

def main():
    X_train, y_train, X_test, y_test = load_data()
    X_train = preprocess_for_ml(X_train)
    X_test = preprocess_for_ml(X_test)

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ SVM Test Accuracy: {acc:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/svm_emotion_model.pkl")
    print("💾 Saved: models/svm_emotion_model.pkl")

if __name__ == "__main__":
    main()
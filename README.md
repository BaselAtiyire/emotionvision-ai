EmotionVision — Facial Emotion Recognition (CNN + Transfer Learning + Explainable AI)

A Computer Vision / Deep Learning project that detects 7 human emotions from facial images using Convolutional Neural Networks and Transfer Learning.
The system includes model training, evaluation, explainability (Grad-CAM), real-time webcam detection, and an interactive Streamlit demo.

This project demonstrates end-to-end AI engineering skills including:

Deep Learning (CNN)

Transfer Learning (EfficientNetV2)

Classical ML Baseline (SVM)

Model evaluation and visualization

Explainable AI (Grad-CAM)

Real-time computer vision (OpenCV)

Interactive ML apps (Streamlit)

                
               Architecture Diagram 
                +-----------------------------+
                |   FER Dataset (.npy files)  |
                | train/test images + labels  |
                +--------------+--------------+
                               |
                               v
                +-----------------------------+
                |     Data Preprocessing      |
                | normalize, reshape, split  |
                +------+----------------------+
                       |
         +-------------+-------------+
         |                           |
         v                           v
+-------------------+      +----------------------+
|   CNN Training    |      |   SVM Baseline       |
| TensorFlow / Keras|      | scikit-learn         |
+---------+---------+      +----------+-----------+
          |                           |
          v                           v
+------------------------+   +------------------------+
| cnn_emotion_model.keras|   | svm_emotion_model.pkl |
+-----------+------------+   +------------------------+
            |
            v
+-----------------------------+
|   Evaluation + Reporting    |
| accuracy/loss/confusion     |
| matrix/classification rep.  |
+-------------+---------------+
              |
   +----------+----------+------------------+
   |                     |                  |
   v                     v                  v
+---------+     +----------------+   +-------------------+
|Streamlit|     |   Grad-CAM     |   |   Webcam Demo     |
| Upload  |     | Explainability |   | Real-time emotion |
| Preview |     | Heatmap overlay|   | detection         |
+---------+     +----------------+   +-------------------+


Project Structure
EmotionVision/
│
├── data/
│   train_images.npy
│   train_labels.npy
│   test_images.npy
│   test_labels.npy
│
├── src/
│   train_cnn.py
│   train_svm.py
│   train_tl.py
│   evaluate.py
│   data_utils.py
│
├── models/
│   cnn_emotion_model.keras
│   efficientnetv2_emotion.keras
│   svm_emotion_model.pkl
│
├── outputs/
│   cnn_accuracy_curve.png
│   cnn_loss_curve.png
│   confusion_matrix_cnn.png
│   gradcam_overlay.png
│
├── assets/
│   webcam_demo_1.png
│   webcam_demo_2.png
│   streamlit_ui_preview.png
│
├── gradcam.py
├── webcam_demo.py
├── appstreamlit_app.py
├── requirements.txt
└── README.md


Models Used
CNN (Baseline Deep Learning Model)

Custom Convolutional Neural Network built using TensorFlow / Keras.

Input:

48 × 48 grayscale images

Architecture:

Conv2D → MaxPool
Conv2D → MaxPool
Conv2D → MaxPool
Flatten
Dense
Softmax

Purpose:

lightweight deep learning baseline

good training speed

SVM Baseline (Classical ML)

Support Vector Machine trained on flattened images.

Libraries:

scikit-learn

Purpose:

classical ML comparison

verify deep learning improvement

Transfer Learning Model

Pretrained EfficientNetV2B0 used for transfer learning.

Process:

48×48 grayscale → 224×224 RGB
ImageNet pretrained backbone
Fine-tuning final layers

Benefits:

stronger feature extraction

better generalization

higher accuracy potential

Results
Model	Input	Test Accuracy	Notes
CNN	48×48 grayscale	XX.XX%	Custom baseline deep learning
SVM	Flattened pixels	XX.XX%	Classical ML baseline
EfficientNetV2	224×224 RGB	XX.XX%	Transfer learning
Training Curves
Accuracy

Loss

Confusion Matrix

Shows classification performance across the 7 emotion categories.

Grad-CAM Explainability

Grad-CAM highlights which facial regions influenced the CNN prediction.

This helps interpret how the model makes decisions.

Webcam Emotion Detection

Real-time emotion detection using OpenCV + TensorFlow.

Pipeline:

Webcam frame
→ Face detection
→ Resize to 48×48
→ CNN prediction
→ Overlay emotion label

Example output:

Run locally:

python webcam_demo.py

Press q to quit.

Streamlit Interactive App

The project also includes an interactive Streamlit dashboard.

Features:

Upload image

Predict emotion

Display class probabilities

Visual interface for model demo

Preview:

Run the app:

streamlit run appstreamlit_app.py
Installation

Clone repository

git clone https://github.com/YOUR_USERNAME/emotionvision.git
cd emotionvision

Create environment

python -m venv cnn_env
cnn_env\Scripts\activate

Install dependencies

pip install -r requirements.txt
Training the Models

Train CNN:

python -m src.train_cnn

Train SVM baseline:

python -m src.train_svm

Train transfer learning model:

python -m src.train_tl
Model Evaluation
python -m src.evaluate

Outputs:

confusion matrix

classification metrics

saved plots

Grad-CAM Visualization

Generate explainability heatmap:

python gradcam.py

Output:

outputs/gradcam_overlay.png
Key Skills Demonstrated

Deep Learning

Computer Vision

Transfer Learning

Explainable AI

Real-time AI systems

Model evaluation

Python ML ecosystem

Interactive AI dashboards

Future Improvements

Potential upgrades:

Real-time Grad-CAM visualization

Face tracking with MediaPipe

Model quantization for edge deployment

Docker containerization

Cloud deployment (AWS / GCP)
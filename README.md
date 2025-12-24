# 🤟 Sign Language Recognition with Mediapipe & Machine Learning

[![Python](https://img.shields.io/badge/python-3.11-blue.svg?style=flat-square)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)  

This project is a **real-time hand gesture recognition system** using **Mediapipe**, **OpenCV**, and **machine learning**. It allows recognizing static sign language gestures via webcam and provides a pipeline to train, evaluate, and save models using **Random Forest** and **K-Nearest Neighbors (KNN)** classifiers.

---

## 📝 Project Overview

The system works in three stages:

1. **Feature Extraction**  
   - Uses Mediapipe Hands to detect 21 hand landmarks per frame.  
   - Normalizes the landmarks to a consistent scale and orientation.

2. **Model Training**  
   - Extracted features are saved into a CSV dataset.  
   - Train classifiers (**Random Forest**, **KNN**) to predict the gesture class.  
   - Save trained models and label encoder for real-time inference.

3. **Real-Time Prediction**  
   - Capture webcam feed and detect hand landmarks in real-time.  
   - Predict gestures using the trained model.  
   - Smooth predictions using a rolling window to reduce noise.  
   - Display the predicted gesture on the video frame.

---

## 💻 Features

- Real-time hand gesture recognition from webcam
- Supports training new datasets of hand gestures
- Normalization of landmarks for rotation and scale invariance
- Multiple classifiers: Random Forest & KNN
- Rolling window prediction to stabilize outputs
- Confusion matrix visualization for model evaluation

---

## 🔧 Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>

2. Install dependencies:
pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib tensorflow seaborn joblib

##🏃‍♂️ Running the Project

1. Feature Extraction & Dataset Preparation

# Run the dataset feature extraction script
python feature_extraction.py
This will generate a CSV file (features.csv) containing normalized landmarks and labels.
git clone <repository-url>
cd <repository-folder>

2. Train Models

python train_models.py

Trains Random Forest and KNN classifiers.
Saves models as sign_rf_model.pkl and sign_knn_model.pkl.
Saves the label encoder as label_encoder.pkl.
Prints accuracy, classification report, and confusion matrix.

3. Real-Time Gesture Recognition

python realtime_prediction.py

Opens a webcam window.
Detects hand gestures in real-time.
Displays the predicted gesture on the screen.
Press q to quit.

##📊 Model Evaluation

Confusion matrices for both Random Forest and KNN are generated using Seaborn heatmaps.

Accuracy and classification reports are displayed in the console.

##👩‍💻 Team / Author

Mariam Aly – mariam.aly.2024@aiu.edu.eg
Catherine Gaballah – catherine.gaballah.2024@aiu.edu.eg
Menna Allah Osama Khalil – mennatallah.khalil.2024@aiu.edu.eg

This project was developed as part of Computer Science & AI coursework at Alamein International University.

##🚀 Future Improvements

Add more gestures and dynamic gesture recognition (motion-based)
Integrate with a GUI using Streamlit for easy testing
Deploy as a web or mobile application
Implement deep learning models (CNNs) for higher accuracy
Allow saving recognized gestures to a text file for communication purposes

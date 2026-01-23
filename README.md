# Sign Language Recognition

This project recognizes hand signs from images and live video using **MediaPipe** and machine learning. It supports both **offline image feature extraction** and **real-time hand sign prediction** using a webcam.

---

## 🖥 Project Structure

Sign-Language-Recognition/
│
├── notebooks/
│ └── sign_language_system.ipynb # Main refactored notebook
├── models/
│ ├── sign_rf_model.pkl # Trained RandomForest model
│ ├── sign_knn_model.pkl # Trained KNN model
│ └── label_encoder.pkl # Label encoder for target labels
├── features.csv # Extracted hand features
├── README.md # This file
└── requirements.txt # Required Python packages


---

## 📘 About the Notebook

The notebook `sign_language_system.ipynb` is organized into **four stages**:

1. **Feature Extraction**
   - Uses MediaPipe to detect hand landmarks from images
   - Normalizes and flattens landmarks into feature vectors
   - Saves features to `features.csv`

2. **Model Training**
   - Loads extracted features
   - Trains **RandomForest** and **KNN** classifiers
   - Saves trained models and label encoder

3. **Evaluation**
   - Computes accuracy and classification reports
   - Plots confusion matrices for both classifiers

4. **Real-Time Prediction**
   - Uses webcam to detect hand signs live
   - Uses a **sliding window** to stabilize predictions
   - Shows predicted sign on the screen

---

## 🛠 Requirements

Install all required packages using:

```bash
pip install -r requirements.txt

Key packages:

mediapipe

opencv-python

numpy

pandas

scikit-learn

joblib

matplotlib

seaborn

---
🚀 How to Run

Feature Extraction and Model Training

Open sign_language_system.ipynb

Run cells in order to extract features, train models, and evaluate them

Real-Time Prediction

Run the final cell for live webcam sign recognition

Press q to quit the webcam window
---
📈 Results

RandomForest Accuracy: ~[0.97]

KNN Accuracy: ~[0.94]

Confusion matrices are plotted inside the notebook for analysis.
---
💡 Notes

Make sure your dataset folder path is correct in the notebook

For real-time prediction, only one hand is supported at a time

Prediction uses a deque window to stabilize results
---
📚 References
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
https://scikit-learn.org/stable/
---
🎯 Contact
Menna Allah Osama
Email:mennatallah.khalil.2024@aiu.edu.eg
GitHub:MennaOsama01
LinkedIn:linkedin.com/in/menna-osama-a01943344

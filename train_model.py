import numpy as np
import librosa
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(file_path):
    data, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    # Use 20 MFCCs to match model input size
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20)
    features = np.mean(mfccs.T, axis=0)
    return features

# -----------------------------
# Load dataset (update path)
# -----------------------------
dataset_path = "dataset"  # folder containing subfolders of emotions
X, y = [], []

for emotion_dir in os.listdir(dataset_path):
    emotion_path = os.path.join(dataset_path, emotion_dir)
    if os.path.isdir(emotion_path):
        for file in os.listdir(emotion_path):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_path, file)
                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(emotion_dir)
                except Exception as e:
                    print("Error:", file_path, e)

X = np.array(X)
y = np.array(y)

# -----------------------------
# Train/test split & model training
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Save model
# -----------------------------
pickle.dump(model, open("emotion_model.pkl", "wb"))
print("Model saved successfully as emotion_model.pkl")

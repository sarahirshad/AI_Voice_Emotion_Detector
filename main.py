# main.py
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Path to audio dataset
DATA_PATH = "data/"
MODEL_PATH = "models/emotion_model.pkl"

# Map RAVDESS emotion codes to labels
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_emotion(filename):
    emotion_code = filename.split("-")[2]
    return EMOTIONS.get(emotion_code, "unknown")

def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # MFCC
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    # Chroma
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    return np.concatenate((mfccs, chroma, mel))

# Extract features and labels
features_list = []
labels = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            emotion = extract_emotion(file)
            try:
                feature = extract_features(file_path)
                features_list.append(feature)
                labels.append(emotion)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

print(f"Total files found: {len(features_list)}")


# Convert to numpy arrays
X = np.array(features_list)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Test single file (optional)
# test_file = "data/test.wav"
# feature = extract_features(test_file).reshape(1, -1)
# pred = le.inverse_transform(clf.predict(feature))
# print(f"Predicted Emotion for {test_file}: {pred[0]}")

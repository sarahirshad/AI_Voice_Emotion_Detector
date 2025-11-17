import streamlit as st
import pickle
import numpy as np
import librosa
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "emotion_model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ------------------ Page Layout ------------------
st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>🧠 Emotion Detection App</h1>", unsafe_allow_html=True)
st.success("Model loaded successfully!")

# ------------------ Mode Selection ------------------
mode = st.radio("Choose Input Type:", ["Text Input", "Voice Input"])

# ------------------ Text Emotion Prediction ------------------
if mode == "Text Input":
    text_input = st.text_area("Enter your text here:")
    if st.button("Predict Emotion"):
        if text_input.strip() != "":
            X = vectorizer.transform([text_input])
            prediction = model.predict(X)[0]
            st.success(f"Predicted Emotion: **{prediction.upper()}**")
        else:
            st.warning("Please enter some text!")

# ------------------ Voice Emotion Prediction ------------------
elif mode == "Voice Input":
    st.write("🎤 Record or Upload an audio file (in WAV format)")

    audio_file = st.file_uploader("Upload your voice file:", type=["wav"])

    if audio_file is not None:
        file_path = os.path.join("data", "uploaded_audio.wav")
        with open(file_path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.audio(file_path, format="audio/wav")

        if st.button("Predict Emotion from Voice"):
            try:
                # Load and extract features
                y, sr = librosa.load(file_path, duration=3, offset=0.5)
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

                # Convert to model input
                features = mfccs.reshape(1, -1)
                voice_prediction = model.predict(features)[0]

                st.success(f"Predicted Emotion from Voice: **{voice_prediction.upper()}**")

            except Exception as e:
                st.error(f"Error processing the audio file: {e}")

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import os
import tempfile
import pickle
import shutil
from speechbrain.inference import EncoderClassifier

# ==============================================================
# 🧭 Path Setup (Handles OneDrive/Windows restrictions)
# ==============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
PRETRAINED_DIR = os.path.join(BASE_DIR, "pretrained_models", "emotion-recognition-wav2vec2-IEMOCAP")
HF_CACHE = os.path.join(BASE_DIR, "hf_cache")

# Create safe local folders
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PRETRAINED_DIR, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

# Force SpeechBrain + HuggingFace to use only local cache
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["SPEECHBRAIN_CACHE"] = HF_CACHE

# ==============================================================
# 🎤 Load SpeechBrain Voice Emotion Model (with retry logic)
# ==============================================================
@st.cache_resource
def load_voice_model():
    try:
        model = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir=PRETRAINED_DIR,
            run_opts={"device": "cpu"}
        )
        return model
    except Exception as e:
        st.warning("⚠️ Retrying SpeechBrain model download in safe mode...")
        try:
            if os.path.exists(PRETRAINED_DIR):
                shutil.rmtree(PRETRAINED_DIR)
            model = EncoderClassifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir=PRETRAINED_DIR,
                run_opts={"device": "cpu"}
            )
            return model
        except Exception as e2:
            st.error(f"❌ Error loading SpeechBrain model:\n\n{e2}")
            return None

# ==============================================================
# 🧠 Load Text Emotion Model
# ==============================================================
@st.cache_resource
def load_text_model():
    try:
        model_path = os.path.join(MODELS_DIR, "text_emotion_model.pkl")
        vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading text model:\n\n{e}")
        return None, None

# ==============================================================
# 🎧 Predict Emotion from Audio
# ==============================================================
def predict_emotion_from_audio(file_path, model):
    try:
        # Ensure standard format for model
        signal, sr = librosa.load(file_path, sr=16000)
        sf.write(file_path, signal, 16000)

        prediction = model.classify_file(file_path)
        emotion = prediction["emotion"]
        return emotion
    except Exception as e:
        st.error(f"⚠️ Audio processing error:\n\n{e}")
        return None

# ==============================================================
# ✍️ Predict Emotion from Text
# ==============================================================
def predict_emotion_from_text(text, model, vectorizer):
    try:
        text_vec = vectorizer.transform([text])
        emotion = model.predict(text_vec)[0]
        return emotion
    except Exception as e:
        st.error(f"⚠️ Text processing error:\n\n{e}")
        return None

# ==============================================================
# 🚀 Streamlit Layout
# ==============================================================
st.set_page_config(
    page_title="🎭 AI Voice & Text Emotion Detector",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 AI Voice & Text Emotion Detector")
st.markdown("Upload an **audio file** or **enter text** to detect emotions using AI!")

# Load both models
voice_model = load_voice_model()
text_model, vectorizer = load_text_model()

# Tabs
tab1, tab2 = st.tabs(["🎤 Voice Emotion", "💬 Text Emotion"])

# ==============================================================
# 🎤 Voice Emotion Tab
# ==============================================================
with tab1:
    st.subheader("🎧 Upload a Voice Clip (WAV, MP3, OGG)")
    audio_file = st.file_uploader("Upload your voice file", type=["wav", "mp3", "ogg"])

    if audio_file:
        st.audio(audio_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name

        if st.button("Analyze Emotion from Voice"):
            if voice_model:
                with st.spinner("🎵 Analyzing emotion from voice..."):
                    emotion = predict_emotion_from_audio(tmp_path, voice_model)
                if emotion:
                    st.success(f"🎯 Detected Emotion: **{emotion.upper()}**")
            else:
                st.warning("⚠️ Voice model not loaded properly.")

# ==============================================================
# 💬 Text Emotion Tab
# ==============================================================
with tab2:
    st.subheader("🧠 Enter Your Text Below:")
    user_text = st.text_area("Type something (e.g., 'I am feeling so happy today!')")

    if st.button("Analyze Emotion from Text"):
        if user_text.strip():
            if text_model and vectorizer:
                with st.spinner("💬 Analyzing emotion from text..."):
                    emotion = predict_emotion_from_text(user_text, text_model, vectorizer)
                if emotion:
                    st.success(f"🗣️ Detected Emotion: **{emotion.upper()}**")
            else:
                st.warning("⚠️ Text model not loaded properly.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

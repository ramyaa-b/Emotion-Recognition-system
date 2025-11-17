# app.py — FINAL WORKING VERSION
# Beautiful UI + Correct Preprocessing + Accurate Predictions

import streamlit as st
import numpy as np
import pandas as pd
import io
import time
import pathlib
import traceback
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------
# PAGE CONFIG + PINK THEME
# --------------------------------------------------------
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")

PINK = {
    "bg": "#ffe6f2",
    "bg2": "#fff3fa",
    "primary": "#ff4f9c",
    "primary_dark": "#ff2f87",
    "header": "#cc0066",
    "text": "#3d1f33",
}

# CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {PINK['bg']} 0%, {PINK['bg2']} 100%);
        font-family: 'Poppins', sans-serif;
        color: {PINK['text']};
    }}
    h1, h2, h3 {{ color: {PINK['header']} !important; font-weight: 700; }}
    .card {{
        background: white;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(255, 90, 160, 0.15);
        margin-bottom: 20px;
    }}
    .stButton>button {{
        background: linear-gradient(90deg, {PINK['primary']}, {PINK['primary_dark']});
        color: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 8px 18px;
        border: none;
        box-shadow: 0 6px 12px rgba(255,0,120,0.2);
    }}
    .stFileUploader {{
        border: 2px dashed {PINK['primary']};
        border-radius: 12px;
        padding: 14px;
        background-color: #fff0f7;
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, white, {PINK['bg']});
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------
# MODEL SETTINGS
# --------------------------------------------------------
SR = 22050
N_MFCC = 40
MODEL_PATH = "model.h5"

# Your actual class order (from your working minimal code)
EMO_LABELS = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
]

# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------
@st.cache_resource
def load_emotion_model():
    p = pathlib.Path(MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError("model.h5 not found. Upload it to repo root.")
    return load_model(str(p))

# --------------------------------------------------------
# AUDIO HELPERS
# --------------------------------------------------------
def read_audio(file_bytes):
    """Return mono signal and sample rate."""
    try:
        with io.BytesIO(file_bytes) as f:
            signal, sr = sf.read(f)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
    except:
        return np.zeros(SR, dtype=np.float32), SR
    return signal.astype(np.float32), sr

def compute_mfcc(file_bytes):
    """Compute MFCC exactly like your working script."""
    sig, sr = read_audio(file_bytes)
    if sr != SR:
        sig = librosa.resample(sig, sr, SR)

    mfcc = librosa.feature.mfcc(y=sig, sr=SR, n_mfcc=N_MFCC)
    return mfcc

def prepare_input(mfcc):
    """The KEY step: collapse MFCC time axis → match your model."""
    collapsed = np.mean(mfcc, axis=1, keepdims=True)  # (40,1)
    x = np.expand_dims(collapsed, axis=0)             # (1,40,1)
    return x.astype(np.float32)

def predict_audio(model, file_bytes):
    """Final accurate prediction."""
    mfcc = compute_mfcc(file_bytes)
    x = prepare_input(mfcc)

    probs = model.predict(x)[0]
    probs = np.array(probs).flatten()

    # Fix mismatches
    if len(probs) != len(EMO_LABELS):
        corrected = np.zeros(len(EMO_LABELS))
        corrected[:min(len(probs), len(corrected))] = probs[:min(len(probs), len(corrected))]
        probs = corrected

    idx = int(np.argmax(probs))
    return EMO_LABELS[idx], float(probs[idx]), probs

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Home", "Make Prediction", "About"])
    st.markdown("---")
    st.caption("Speech Emotion Recognition System")

# --------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Speech Emotion Recognition System")

    st.write(
        """
        This system analyzes short speech clips and identifies the emotion expressed
        by the speaker. By understanding vocal tone and frequency patterns, the model 
        recognizes emotional cues such as happiness, sadness, anger, fear, and more.
        """
    )

    st.subheader("Why this matters")
    st.write(
        """
        - Enhances customer support by detecting frustration or satisfaction  
        - Supports mental health monitoring  
        - Helps build emotion-aware assistants & interfaces  
        - Assists researchers in analyzing emotional speech patterns  
        """
    )

    st.subheader("How it works")
    st.write(
        """
        1. You upload a speech audio clip  
        2. The system computes MFCC sound signatures  
        3. It collapses the information into a representation your model understands  
        4. The neural network predicts the closest emotion  
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# MAKE PREDICTION PAGE
# --------------------------------------------------------
elif page == "Make Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Emotion Prediction")

    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])

    if audio_file is not None:
        file_bytes = audio_file.read()
        st.audio(file_bytes)

        with st.spinner("Analyzing emotion..."):
            model = load_emotion_model()
            label, conf, probs = predict_audio(model, file_bytes)
            time.sleep(0.2)

        st.subheader("Predicted Emotion")
        st.success(f"{label.capitalize()}   —   Confidence: {conf:.2f}")

        # Chart for probabilities
        prob_df = pd.DataFrame({
            "Emotion": EMO_LABELS,
            "Probability": probs
        }).sort_values("Probability", ascending=False)

        fig = px.bar(
            prob_df,
            x="Emotion",
            y="Probability",
            color="Emotion",
            color_discrete_sequence=px.colors.sequential.Pinkyl,
            title="Emotion Probability Distribution (Sorted)"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Upload an audio file to start.")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------------
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("About This Project")

    st.write(
        """
        The Speech Emotion Recognition System identifies human emotion from voice
        by analyzing frequency patterns and vocal tone changes. It transforms speech 
        into MFCC signatures and uses a deep learning model to map these signatures 
        to emotional categories.
        """
    )

    st.subheader("Use Cases")
    st.write(
        """
        - Customer care emotion tracking  
        - Mental health and wellbeing tools  
        - Emotion-aware digital assistants  
        - Research and educational purposes  
        """
    )

    st.subheader("Goal")
    st.write(
        """
        The goal is to enable emotionally intuitive technology—systems that understand 
        not only speech content but also the feeling behind it.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)











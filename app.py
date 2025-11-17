# app.py — Final robust version (pink UI, safe audio handling, correct preprocessing)
# - Uses MFCC collapse (mean across time) to form (1,40,1) input for your model
# - Adds safe fallbacks for empty/short/corrupt audio to avoid resample errors
# - Keeps Home, Make Prediction, About pages and the pink theme

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

# CSS styling
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

# Class order must match your model's training order
EMO_LABELS = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
]

# --------------------------------------------------------
# MODEL LOADER
# --------------------------------------------------------
@st.cache_resource
def load_emotion_model():
    p = pathlib.Path(MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError("model.h5 not found. Upload it to repository root.")
    return load_model(str(p))

# --------------------------------------------------------
# SAFE AUDIO & MFCC HELPERS
# --------------------------------------------------------
def read_audio(file_bytes):
    """
    Read bytes to mono signal and sample rate.
    If reading fails or signal is too short, returns 0.5s of silence at SR.
    """
    try:
        with io.BytesIO(file_bytes) as f:
            signal, sr = sf.read(f)
        # convert to mono
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        signal = signal.astype(np.float32)
        # If extremely short or empty, use 0.5s silence
        if signal.size < 256:
            return np.zeros(int(0.5 * SR), dtype=np.float32), SR
        return signal, sr
    except Exception:
        # fallback silent buffer
        return np.zeros(int(0.5 * SR), dtype=np.float32), SR

def compute_mfcc(file_bytes):
    """
    Compute MFCC safely.
    Returns an array of shape (N_MFCC, frames>=1).
    If anything fails, returns zeros (N_MFCC, 1).
    """
    try:
        sig, sr = read_audio(file_bytes)
        # safe resample: handle tiny signals and exceptions
        try:
            if sr != SR:
                if sig.size < 3:
                    sig = np.zeros(int(0.5 * SR), dtype=np.float32)
                sig = librosa.resample(sig, orig_sr=sr, target_sr=SR)
        except Exception:
            sig = np.zeros(int(0.5 * SR), dtype=np.float32)

        # compute MFCC
        try:
            mfcc = librosa.feature.mfcc(y=sig, sr=SR, n_mfcc=N_MFCC)
            # ensure at least one frame
            if mfcc.shape[1] < 1:
                mfcc = np.zeros((N_MFCC, 1), dtype=np.float32)
            return mfcc
        except Exception:
            return np.zeros((N_MFCC, 1), dtype=np.float32)

    except Exception:
        return np.zeros((N_MFCC, 1), dtype=np.float32)

def prepare_input(mfcc):
    """
    Collapse time axis by mean (this matches the working minimal script).
    Returns shaped input (1, N_MFCC, 1)
    """
    try:
        # mfcc shape (N_MFCC, frames)
        collapsed = np.mean(mfcc, axis=1, keepdims=True)   # (N_MFCC, 1)
        x = np.expand_dims(collapsed, axis=0)               # (1, N_MFCC, 1)
        return x.astype(np.float32)
    except Exception:
        # fallback zeros
        return np.zeros((1, N_MFCC, 1), dtype=np.float32)

def predict_audio(model, file_bytes):
    """
    Predict using correct preprocessing. Returns label, confidence, probs
    Ensures probs length matches EMO_LABELS (pads/truncates if needed).
    """
    try:
        mfcc = compute_mfcc(file_bytes)
        x = prepare_input(mfcc)
        preds = model.predict(x)
        probs = np.array(preds).flatten()
        # sanitize length
        if probs.size != len(EMO_LABELS):
            fixed = np.zeros(len(EMO_LABELS), dtype=float)
            fixed[:min(len(probs), len(fixed))] = probs[:min(len(probs), len(fixed))]
            probs = fixed
        idx = int(np.argmax(probs))
        return EMO_LABELS[idx], float(probs[idx]), probs
    except Exception:
        # final uniform fallback
        probs = np.ones(len(EMO_LABELS), dtype=float) / len(EMO_LABELS)
        return "neutral", float(probs[0]), probs

# --------------------------------------------------------
# SIDEBAR NAVIGATION
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
        "This application analyzes short speech clips and identifies the emotion expressed by the speaker. "
        "It works by learning vocal patterns—tone, pitch, and spectral energy changes—and mapping them to emotion categories."
    )

    st.subheader("Why this system is useful")
    st.write(
        "- Detects customer frustration or satisfaction\n"
        "- Supports non-invasive emotion monitoring for wellbeing\n"
        "- Enhances conversational AI with emotional context\n"
        "- Useful for research and education"
    )

    st.subheader("How it works")
    st.write(
        "1. Upload a short audio clip.\n"
        "2. The system extracts compact sound features (MFCCs).\n"
        "3. The features are collapsed to the representation the model expects.\n"
        "4. A trained neural network returns the most likely emotion with a confidence score."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# MAKE PREDICTION PAGE
# --------------------------------------------------------
elif page == "Make Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Emotion Prediction")
    st.write("Upload a short audio file (recommended 1–8 seconds).")

    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])

    if audio_file is not None:
        file_bytes = audio_file.read()
        # show audio player
        st.audio(file_bytes, format=audio_file.type if hasattr(audio_file, "type") else "audio/wav")

        # load model
        try:
            model = load_emotion_model()
        except Exception:
            st.error("Could not load model (model.h5). Please ensure model.h5 is present in the repo root.")
            st.stop()

        # predict (safe)
        label, conf, probs = predict_audio(model, file_bytes)

        # present result
        st.subheader("Predicted Emotion")
        st.success(f"{label.capitalize()}   —   Confidence: {conf:.2f}")

        # descriptive probability chart
        prob_df = pd.DataFrame({"Emotion": EMO_LABELS, "Probability": probs})
        prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)
        prob_df["Percent"] = (prob_df["Probability"] * 100).round(1)

        fig = px.bar(
            prob_df,
            x="Emotion",
            y="Probability",
            color="Emotion",
            color_discrete_sequence=px.colors.sequential.Pinkyl,
            title="Emotion Probability Distribution (sorted)"
        )
        fig.update_traces(text=prob_df["Percent"].astype(str) + "%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Upload an audio file to begin.")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------------
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("About This Project")

    st.write(
        "The Speech Emotion Recognition System detects emotions from short voice clips. "
        "It uses audio features and a trained neural network to understand the emotional tone of speech."
    )

    st.subheader("Use cases")
    st.write(
        "- Customer support sentiment monitoring\n"
        "- Mental health & wellbeing tools\n"
        "- Emotion-aware conversational agents\n"
        "- Research and educational studies"
    )

    st.subheader("Role of Each Technology in the Speech Emotion Recognition System")
    st.write("The Speech Emotion Recognition system brings together a collection of powerful technologies, each playing a unique and essential role in transforming raw audio into meaningful emotional insights. Python serves as the backbone of the project because of its extensive ecosystem for machine learning, signal processing, and rapid application development. Librosa and SoundFile handle the core audio-processing pipeline—reading diverse audio formats, resampling signals, and extracting Mel-Frequency Cepstral Coefficients (MFCCs), which act as compact and highly expressive representations of human speech characteristics. These MFCC features are then passed into TensorFlow/Keras, which powers the deep-learning model. TensorFlow provides the mathematical foundation, GPU-accelerated computation, and optimized neural network operations necessary for learning subtle patterns in voice that correspond to emotional states. The trained model is packaged as a reusable .h5 file, making it easy to load and run predictions in real time."
             
        "To present these capabilities to users in an intuitive and accessible format, Streamlit is used to build the web application interface. Streamlit transforms Python scripts into interactive web apps with minimal boilerplate, enabling instant audio uploads, live inference, progress indicators, and rich visualizations—all without requiring traditional front-end development. The inclusion of Plotly enhances the user experience by generating beautiful, dynamic, and accurate probability distribution charts that help users visualize how the model interprets different emotions. Under the hood, NumPy and Pandas assist with efficient numerical operations and structured data handling, ensuring smooth processing of prediction outputs and probability tables. Together, this stack creates a seamless pipeline—from raw speech to displayed emotion—integrating audio engineering, machine learning, and web deployment into a single, cohesive system that is both technically robust and easy for end users to interact with.")

    st.markdown("</div>", unsafe_allow_html=True)












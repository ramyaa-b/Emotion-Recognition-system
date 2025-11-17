# app.py

import streamlit as st
import numpy as np
import pandas as pd
import io
import time
import pathlib

import librosa
import soundfile as sf

from tensorflow.keras.models import load_model

import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------------------
# BASIC CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🔊",
    layout="wide"
)

PINK = {
    "bg_light": "#ffe6f1",
    "bg": "#ffd0e0",
    "primary": "#ff4f9c",
    "primary_dark": "#ff2f87",
    "card": "white",
    "text": "#4a2340",
}

MODEL_PATH = "model.h5"
EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

SR = 22050
N_MFCC = 40
MAX_LEN = 173


# --------------------------------------------------------------
# CUSTOM CSS — PINK THEME
# --------------------------------------------------------------
st.markdown(
    f"""
    <style>

    .stApp {{
        background: linear-gradient(180deg, {PINK['bg_light']} 0%, #ffffff 100%);
        color: {PINK['text']};
        font-family: 'Poppins', sans-serif;
    }}

    .card {{
        background: {PINK['card']};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 4px 15px rgba(255, 79, 156, 0.12);
        margin-bottom: 22px;
    }}

    .stButton>button {{
        background: linear-gradient(90deg, {PINK['primary']} 0%, {PINK['primary_dark']} 100%);
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border: none;
        box-shadow: 0px 4px 10px rgba(255, 47, 135, 0.25);
    }}

    .stButton>button:hover {{
        opacity: 0.92;
    }}

    .stRadio>label {{
        font-size: 17px;
        font-weight: 500;
        color: {PINK['text']};
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #ffffff 0%, {PINK['bg_light']} 100%);
    }}

    .stFileUploader {{
        border: 2px dashed {PINK['primary']};
        border-radius: 10px;
        padding: 15px;
        background-color: #fff5fa;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------------------
# MODEL + AUDIO HELPERS
# --------------------------------------------------------------
@st.cache_resource
def load_emotion_model():
    if not pathlib.Path(MODEL_PATH).exists():
        raise FileNotFoundError("model.h5 not found in root directory.")
    return load_model(MODEL_PATH)


def read_audio(audio_bytes):
    with io.BytesIO(audio_bytes) as f:
        signal, sr = sf.read(f)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    return signal.astype(np.float32), sr


def extract_mfcc(audio_bytes):
    signal, sr = read_audio(audio_bytes)
    if sr != SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=SR)

    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)

    if mfcc.shape[1] < MAX_LEN:
        pad_len = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_len)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc.astype(np.float32)


def predict_audio(model, audio_bytes):
    x = extract_mfcc(audio_bytes)
    probs = model.predict(x)[0]
    idx = np.argmax(probs)
    return EMO_LABELS[idx], float(probs[idx]), probs


# --------------------------------------------------------------
# SIDEBAR — Minimal Navigation
# --------------------------------------------------------------
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Home", "Make Prediction", "About"])
    st.markdown("---")
    st.caption("Authors")    
    st.caption("Ramyaa Balasubramanian")
    st.caption("Shashin Vathode")
    st.caption("Shreya Chaudhari")
    st.caption("---")
    st.caption("Speech Emotion Recognition System")


# --------------------------------------------------------------
# HOME PAGE — Professional + Eye Catching
# --------------------------------------------------------------
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Speech Emotion Recognition System")

    st.markdown(
        """
        This system uses cutting-edge machine learning to understand human emotion **purely through voice**.  
        By analysing tone, frequency patterns, and vocal energy, the model identifies the **dominant emotion** expressed in a speech clip.

        The goal is to make **emotion-aware technology** accessible to researchers, developers, psychologists, and innovation teams.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Why this system is useful")
        st.write(
            """
            - Enables emotion-aware applications  
            - Supports behavioral & psychological analysis  
            - Helps in improving AI conversations  
            - Useful for call-center monitoring  
            - Valuable for user research & product design  
            - Enhances assistive technologies  
            """
        )

    with col2:
        st.subheader("How the model works")
        st.write(
            """
            1. You upload a short speech audio file.  
            2. The system extracts MFCC features representing vocal patterns.  
            3. The Keras deep learning model processes the features.  
            4. It outputs the most likely emotion along with confidence levels.  
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:17px; font-weight:500;'>Try the prediction tool to experience emotion analysis in real time.</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------
# MAKE PREDICTION PAGE — Minimalistic, Clean, Pink
# --------------------------------------------------------------
elif page == "Make Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Emotion Prediction")

    st.write("Upload a short audio clip (wav, mp3, or m4a) to detect the emotion expressed in the speech.")

    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])

    if audio_file:
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format=audio_file.type)

        with st.spinner("Analyzing audio..."):
            model = load_emotion_model()
            label, conf, probs = predict_audio(model, audio_bytes)
            time.sleep(0.4)

        st.subheader("Prediction Result")
        st.success(f"Emotion: {label}  |  Confidence: {conf:.2f}")

        prob_df = pd.DataFrame({
            "emotion": EMO_LABELS,
            "probability": probs
        })

        fig = px.bar(
            prob_df,
            x="emotion",
            y="probability",
            color="emotion",
            title="Emotion Probability Distribution",
            color_discrete_sequence=px.colors.sequential.Pinkyl
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------
# ABOUT PAGE — Detailed, Professional
# --------------------------------------------------------------
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("About This Project")

    st.write(
        """
        The Speech Emotion Recognition System is a machine learning application designed to identify 
        human emotion from voice recordings. By processing the acoustic features of speech, the system 
        determines the emotional tone—ranging from happiness and sadness to fear, anger, or neutrality.
        """
    )

    st.subheader("Technology Used")
    st.write(
        """
        - Keras + TensorFlow (model)  
        - MFCC extraction using Librosa  
        - Streamlit for UI deployment  
        - Plotly for visual analytics  
        - SoundFile for secure audio handling  
        """
    )

    st.subheader("How It Works")
    st.write(
        """
        1. Audio is uploaded by the user.  
        2. The system transforms it into MFCC features, a compact representation of sound energy variations.  
        3. The features are fed into a trained deep learning model.  
        4. A probability distribution across predefined emotion classes is generated.  
        5. The highest-confidence label is presented as the predicted emotion.  
        """
    )

    st.subheader("Authors")
    st.write(
        """
        - Ramyaa Balasubramanian  
        - Shashin Vathode  
        - Shreya Chaudhari  
        """
    )

    st.subheader("Purpose & Applications")
    st.write(
        """
        This project demonstrates how audio processing and neural networks can be used to extract emotional 
        information from voice. It can support several domains including research, healthcare, customer support, 
        digital assistants, and human-computer interaction studies.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)





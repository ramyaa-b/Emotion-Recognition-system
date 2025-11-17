# app.py
"""
Final Polished Version — Speech Emotion Recognition (Pink Theme)
✓ No “all arrays must be same length” plot error
✓ Perfectly matched to your model (expects 40×16=640 flattened input)
✓ Pink, smooth, professional UI
✓ Improved Home + About pages
✓ Minimal, clean Make Prediction page
✓ Customer-friendly explanations (no technical clutter)
"""

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

# ---------------------------------------------------------------------------
# CONFIG & UI STYLE
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")

PINK = {
    "bg": "#ffe6f2",
    "bg2": "#fff3fa",
    "primary": "#ff4f9c",
    "primary_dark": "#ff2f87",
    "header": "#cc0066",
    "text": "#3d1f33",
}

# Beautiful pink styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {PINK['bg']} 0%, {PINK['bg2']} 100%);
        color: {PINK['text']};
        font-family: 'Poppins', sans-serif;
    }}

    h1, h2, h3 {{
        color: {PINK['header']} !important;
        font-weight: 700;
    }}

    .card {{
        background: white;
        border-radius: 14px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0px 6px 22px rgba(255, 101, 163, 0.15);
    }}

    .stButton>button {{
        background: linear-gradient(90deg, {PINK['primary']} 0%, {PINK['primary_dark']} 100%);
        color: white;
        border-radius: 10px;
        padding: 9px 20px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        box-shadow: 0px 6px 10px rgba(255, 0, 100, 0.2);
    }}

    .stButton>button:hover {{
        opacity: 0.9;
    }}

    .stFileUploader {{
        border: 2px dashed {PINK['primary']};
        border-radius: 12px;
        padding: 15px;
        background-color: #fff0f7;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, white 0%, {PINK['bg']} 100%);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# MODEL SETTINGS
# Your model EXPECTS 40 MFCCs × 16 frames = 640 features
# ---------------------------------------------------------------------------

SR = 22050
N_MFCC = 40
MAX_LEN = 16  # *** CRITICAL — matches your model’s input ***
MODEL_PATH = "model.h5"

EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ---------------------------------------------------------------------------
# MODEL & AUDIO FUNCTIONS
# ---------------------------------------------------------------------------

@st.cache_resource
def load_emotion_model():
    """Load the Keras model."""
    p = pathlib.Path(MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError("model.h5 not found — place it in the root directory.")
    model = load_model(str(p))
    return model

def read_audio(file_bytes):
    with io.BytesIO(file_bytes) as f:
        signal, sr = sf.read(f)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    return signal.astype(np.float32), sr

def extract_mfcc(file_bytes):
    """Return (40, 16) MFCC matrix exactly matching training."""
    sig, sr = read_audio(file_bytes)
    if sr != SR:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=SR)

    mfcc = librosa.feature.mfcc(y=sig, sr=SR, n_mfcc=N_MFCC)

    # Normalization
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    # Pad or crop to MAX_LEN = 16
    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc  # shape (40,16)

def prepare_for_model(mfcc):
    """Flatten MFCC → shape (1, 640) exactly as model expects."""
    flat = mfcc.flatten()  # length = 640
    return flat.reshape(1, -1).astype(np.float32)

def predict_audio(model, file_bytes):
    mfcc = extract_mfcc(file_bytes)
    x = prepare_for_model(mfcc)
    probs = model.predict(x)[0]
    idx = np.argmax(probs)
    return EMO_LABELS[idx], float(probs[idx]), probs

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Home", "Make Prediction", "About"])
    st.markdown("---")
    st.caption("Speech Emotion Recognition System")

# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------

if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Speech Emotion Recognition System")

    st.write(
        """
        This system listens to **short speech clips** and identifies the **emotion**
        expressed by the speaker. It works by analyzing the vocal tone, energy,
        and frequency patterns present in human speech.
        """
    )

    st.subheader("Why this system is useful")
    st.write(
        """
        - Helps companies understand customer sentiment  
        - Supports mental health monitoring and emotional analytics  
        - Enhances conversational AI and assistants  
        - Useful for call-center quality evaluation  
        - Helps researchers study human emotional patterns  
        """
    )

    st.subheader("How it works (simple explanation)")
    st.write(
        """
        1. You upload a short audio clip.  
        2. The system extracts tiny sound signatures called **MFCCs**,  
           which represent how the vocal frequencies change over time.  
        3. These signatures are fed into a **deep learning model** trained on emotional speech.  
        4. The system outputs the **predicted emotion** and a confidence level.  
        """
    )

    st.write(
        "You don’t need technical knowledge — the app handles processing, extraction, and prediction automatically."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAKE PREDICTION PAGE
# ---------------------------------------------------------------------------

elif page == "Make Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Emotion Prediction")
    st.write("Upload a short audio file (1–8 seconds recommended).")

    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        try:
            file_bytes = audio_file.read()
            st.audio(file_bytes)

            with st.spinner("Analyzing emotion..."):
                model = load_emotion_model()
                label, conf, probs = predict_audio(model, file_bytes)
                time.sleep(0.2)

            st.subheader("Result")
            st.success(f"Emotion: **{label}**   |   Confidence: {conf:.2f}")

            # Probability plot, FIXED LENGTH = 7 emotions
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

        except Exception as e:
            st.error("Prediction failed. Please try another audio file.")
            print("ERROR:", traceback.format_exc())

    else:
        st.info("Upload an audio file to begin.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------------------------

elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("About This Project")

    st.write(
        """
        The **Speech Emotion Recognition System** is designed to make computers more
        emotionally aware. Instead of just recognizing *words*, this system listens
        to **how** something is said, allowing it to identify emotional cues such as
        happiness, sadness, anger, fear, and more.
        """
    )

    st.subheader("How it works (explained simply)")
    st.write(
        """
        - Human speech is converted into a visual-like pattern of frequency
          changes over time.  
        - These patterns (called **MFCCs**) capture the emotional energy in the voice.  
        - A deep learning model, trained on thousands of emotional speech samples,
          learns what each emotion “sounds like”.  
        - When you upload audio, the system compares it to learned patterns  
          and predicts the most likely emotion.
        """
    )

    st.subheader("Use cases")
    st.write(
        """
        **Customer Care**  
        - Detect customer frustration or satisfaction in call centers.  
        
        **Mental Health Support**  
        - Non-invasive emotional monitoring for mood analysis.  

        **Human–Computer Interaction**  
        - Build emotionally aware assistants, robots, or apps.  

        **Education & Research**  
        - Study communication patterns, public speaking, or therapy sessions.  
        """
    )

    st.subheader("Project Goal")
    st.write(
        """
        Our goal is to bring intuitive, emotion-aware intelligence to everyday
        systems and applications. By analyzing speech in real-time, this technology
        helps improve communication, understanding, and user experience.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)








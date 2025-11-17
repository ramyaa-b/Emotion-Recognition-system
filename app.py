# app.py
"""
Speech Emotion Recognition — Streamlit App (Pink theme, corrected)
- This version adapts MFCC input shapes to match the loaded Keras model.
- Place your trained Keras model as `model.h5` in the repo root.
- Authors: Ramyaa Balasubramanian, Shashin Vathode, Shreya Chaudhari
"""

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

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon=None,
    layout="wide"
)

# Pink palette
PINK = {
    "bg_light": "#fff0f6",
    "bg": "#ffe6f1",
    "primary": "#ff4f9c",
    "primary_dark": "#ff2f87",
    "card": "#ffffff",
    "text": "#331a2b",
}

MODEL_PATH = "model.h5"
# Ensure this matches the order used during training
EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Feature extraction defaults (adjust if your training used different values)
SR = 22050
N_MFCC = 40
MAX_LEN = 173

# -----------------------
# CSS styling
# -----------------------
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
        padding: 18px;
        box-shadow: 0px 6px 18px rgba(255, 79, 156, 0.08);
        margin-bottom: 18px;
    }}
    .stButton>button {{
        background: linear-gradient(90deg, {PINK['primary']} 0%, {PINK['primary_dark']} 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 8px 16px;
        box-shadow: 0px 4px 10px rgba(255, 47, 135, 0.14);
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #ffffff 0%, {PINK['bg_light']} 100%);
    }}
    .stFileUploader {{
        border: 2px dashed {PINK['primary']};
        border-radius: 10px;
        padding: 12px;
        background-color: #fff7fb;
    }}
    .dataframe tbody tr:hover {{
        background-color: #ffe9f4;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# MODEL LOADING & ADAPTIVE PREDICTION
# -----------------------
@st.cache_resource
def load_emotion_model(path: str = MODEL_PATH):
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file '{path}' not found. Place your Keras model (model.h5) in the repo root.")
    model = load_model(str(p))
    return model

def read_audio_bytes(audio_bytes: bytes):
    """Read audio bytes using soundfile and return mono signal and sample rate."""
    with io.BytesIO(audio_bytes) as fh:
        data, file_sr = sf.read(fh)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), file_sr

def compute_mfcc_base(audio_bytes: bytes, sr_target=SR, n_mfcc=N_MFCC, max_len=MAX_LEN):
    """
    Compute base MFCC array with shape (n_mfcc, max_len).
    """
    sig, sr = read_audio_bytes(audio_bytes)
    if sr != sr_target:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=sr_target)
    mfcc = librosa.feature.mfcc(y=sig, sr=sr_target, n_mfcc=n_mfcc)
    # normalize (match training normalization if different)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    # pad/truncate
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc  # shape (n_mfcc, max_len)

def predict_audio_adaptive(model, audio_bytes: bytes):
    """
    Adapt MFCC to the model's expected input shape and return (label, confidence, probs).
    This function will try common axis orders and flattening to match the model.
    If prediction fails, it raises a helpful ValueError with shapes.
    """
    base = compute_mfcc_base(audio_bytes)  # (n_mfcc, max_len)
    input_shape = model.input_shape  # e.g. (None, 40, 173, 1) or (None, 6940)
    expected = list(input_shape)[1:]  # remove batch dim

    x = None

    try:
        # Case: expects 3 dims (e.g. (n_mfcc, max_len, channels))
        if len(expected) == 3:
            a, b, c = expected
            # if channel dimension is 1 or 3
            if c in (1, 3):
                # check if ordering matches (n_mfcc, max_len, ch)
                if a == base.shape[0] and b == base.shape[1]:
                    arr = base[..., np.newaxis]  # (n_mfcc, max_len, 1)
                # check if model expects (max_len, n_mfcc, ch)
                elif a == base.shape[1] and b == base.shape[0]:
                    arr = base.T[..., np.newaxis]  # (max_len, n_mfcc, 1)
                else:
                    # attempt best match by assuming (n_mfcc, max_len, 1)
                    arr = base[..., np.newaxis]
                x = np.expand_dims(arr, axis=0).astype(np.float32)

        # Case: expects 2 dims (e.g. (n_mfcc, max_len) or flattened shape (n_features,))
        elif len(expected) == 2:
            e0, e1 = expected
            if (e0 == base.shape[0] and e1 == base.shape[1]):
                x = np.expand_dims(base, axis=0).astype(np.float32)  # (1, n_mfcc, max_len)
            elif (e0 == base.shape[1] and e1 == base.shape[0]):
                x = np.expand_dims(base.T, axis=0).astype(np.float32)
            else:
                # fallback flatten
                x = base.flatten()[np.newaxis, :].astype(np.float32)

        # Case: flattened (1D), e.g. (N,)
        elif len(expected) == 1:
            flat = base.flatten()[np.newaxis, :].astype(np.float32)
            x = flat

        # Other unexpected dims -> fallback to flattened vector
        else:
            x = base.flatten()[np.newaxis, :].astype(np.float32)

        # final ensure float32
        x = x.astype(np.float32)

        # predict
        probs = model.predict(x)
        # normalize output shape
        if isinstance(probs, list):
            probs = np.array(probs)
        probs = np.array(probs)
        if probs.ndim == 2:
            probs = probs[0]
        else:
            probs = probs.flatten()
        idx = int(np.argmax(probs))
        label = EMO_LABELS[idx] if idx < len(EMO_LABELS) else f"Class {idx}"
        return label, float(probs[idx]), probs

    except Exception as e:
        prepared_shape = x.shape if 'x' in locals() and x is not None else None
        raise ValueError(
            f"Model prediction failed. Prepared input shape: {prepared_shape}. "
            f"Model expected: {model.input_shape}. Internal error: {e}"
        ) from e

# -----------------------
# SIDEBAR (minimal nav)
# -----------------------
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Home", "Make Prediction", "About"])
    st.markdown("---")
    st.caption("Authors")
    st.caption("Ramyaa Balasubramanian")
    st.caption("Shashin Vathode")
    st.caption("Shreya Chaudhari")
    st.markdown("---")
    st.caption("Speech Emotion Recognition System")

# -----------------------
# PAGES
# -----------------------
# HOME
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Speech Emotion Recognition System")
    st.write(
        "A focused tool that analyzes short voice clips and identifies the dominant emotion. "
        "The system extracts MFCC features and uses a trained Keras model to return a probability distribution over emotions."
    )

    st.markdown("")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Why this system is useful")
        st.write(
            "- Enables emotion-aware consumer analytics and user research\n"
            "- Supports mental health and behavioral studies\n"
            "- Improves conversational AI and UX through emotional context\n"
            "- Useful for call-center analytics and quality assurance\n"
        )

    with col2:
        st.subheader("How the model works (basic)")
        st.write(
            "1. Upload a short audio clip.\n"
            "2. The app computes MFCCs (a compact representation of the sound).\n"
            "3. The MFCCs feed into a Keras neural network trained to recognize emotion patterns.\n"
            "4. The app displays the top emotion and the full probability distribution."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# MAKE PREDICTION (minimal)
elif page == "Make Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Emotion Prediction")

    st.write("Upload a short audio clip (wav, mp3, or m4a). Recommended length: 1–8 seconds.")

    audio_file = st.file_uploader("", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        try:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format=audio_file.type if hasattr(audio_file, "type") else "audio/wav")

            with st.spinner("Analyzing..."):
                model = load_emotion_model()
                label, conf, probs = predict_audio_adaptive(model, audio_bytes)
                time.sleep(0.25)

            st.subheader("Prediction")
            # High-confidence vs low-confidence display
            if conf >= 0.6:
                st.success(f"Emotion: {label}   |   Confidence: {conf:.2f}")
            else:
                st.warning(f"Emotion: {label}   |   Confidence: {conf:.2f} (low confidence)")

            # Plot probabilities
            prob_df = pd.DataFrame({"emotion": EMO_LABELS, "probability": [float(p) for p in probs]})
            fig = px.bar(prob_df, x="emotion", y="probability", color="emotion", title="Probability distribution",
                         color_discrete_sequence=px.colors.sequential.Pinkyl)
            fig.update_layout(showlegend=False, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            # Allow download
            csv_bytes = prob_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download probabilities (CSV)", data=csv_bytes, file_name="probabilities.csv", mime="text/csv")

        except FileNotFoundError as fnf:
            st.error(str(fnf))
        except ValueError as ve:
            # show helpful message with shapes
            st.error(f"Prediction error: {ve}")
        except Exception as e:
            st.error(f"Unexpected error during prediction: {e}")

    else:
        st.info("Upload an audio file to get a prediction.")

    st.markdown("</div>", unsafe_allow_html=True)

# ABOUT
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("About This Project")

    st.write(
        "The Speech Emotion Recognition System uses acoustic features extracted from short audio clips to infer a speaker's emotion. "
        "It is intended for research, prototyping, and demonstration of emotion-aware speech analytics."
    )

    st.subheader("Technology")
    st.write(
        "- Python 3.12\n"
        "- Keras / TensorFlow (model)\n"
        "- Librosa & SoundFile (audio processing)\n"
        "- Streamlit (UI)\n"
        "- Plotly (visualization)\n"
    )

    st.subheader("How it works (concise)")
    st.write(
        "1. A short voice clip is uploaded.\n"
        "2. The app computes MFCCs, a compact representation capturing spectral patterns.\n"
        "3. A trained neural network predicts emotion probabilities.\n"
        "4. The top emotion is displayed along with confidence and the full distribution."
    )

    st.subheader("Creators")
    st.write("Ramyaa Balasubramanian")
    st.write("Shashin Vathode")
    st.write("Shreya Chaudhari")

    st.subheader("Limitations & notes")
    st.write(
        "- Best results on short, clear, single-speaker audio.\n"
        "- May be less accurate on noisy, multi-speaker, or out-of-domain audio.\n"
        "- Ensure the model.h5 corresponds to the MFCC settings used here (N_MFCC, MAX_LEN)."
    )

    st.markdown("</div>", unsafe_allow_html=True)






# app.py
"""
Speech Emotion Recognition — corrected for your model's input shape.
- This version uses MAX_LEN = 16 (it was likely 173 before).
- The code inspects the model input shape and adapts the MFCC accordingly.
- Place your trained Keras model as `model.h5` in the repo root.
- Authors: Ramyaa Balasubramanian, Shashin Vathode, Shreya Chaudhari
"""

import streamlit as st
import numpy as np
import pandas as pd
import io
import time
import pathlib
import traceback
import logging

import librosa
import soundfile as sf

from tensorflow.keras.models import load_model

import plotly.express as px

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")

# Pink palette (styling)
PINK = {
    "bg_light": "#fff0f6",
    "primary": "#ff4f9c",
    "primary_dark": "#ff2f87",
    "card": "#ffffff",
    "text": "#331a2b",
}

# IMPORTANT: match model training parameters
SR = 22050
N_MFCC = 40
MAX_LEN = 16   # <- CRITICAL: your model expects 40 * 16 = 640 flattened features

MODEL_PATH = "model.h5"
EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# CSS
st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, {PINK['bg_light']} 0%, #ffffff 100%); color: {PINK['text']}; font-family: 'Poppins', sans-serif; }}
    .card {{ background: {PINK['card']}; border-radius: 12px; padding: 18px; box-shadow: 0px 6px 18px rgba(255, 79, 156, 0.08); margin-bottom: 18px; }}
    .stButton>button {{ background: linear-gradient(90deg, {PINK['primary']} 0%, {PINK['primary_dark']} 100%); color: white; border-radius: 10px; font-weight: 600; padding: 8px 16px; }}
    .stFileUploader {{ border: 2px dashed {PINK['primary']}; border-radius: 10px; padding: 12px; background-color: #fff7fb; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# UTILITIES
# -----------------------
@st.cache_resource
def load_emotion_model(path: str = MODEL_PATH):
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file '{path}' not found. Place your Keras model (model.h5) in the repo root.")
    model = load_model(str(p))
    # Print model input shape to logs so you can verify (Streamlit logs / console)
    try:
        print("MODEL INPUT SHAPE:", model.input_shape)
    except Exception:
        pass
    return model

def read_audio_bytes(audio_bytes: bytes):
    with io.BytesIO(audio_bytes) as fh:
        data, file_sr = sf.read(fh)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), file_sr

def compute_mfcc_base(audio_bytes: bytes, sr_target=SR, n_mfcc=N_MFCC, max_len=MAX_LEN):
    """
    Compute MFCC array with shape (n_mfcc, max_len).
    """
    sig, sr = read_audio_bytes(audio_bytes)
    if sr != sr_target:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=sr_target)
    mfcc = librosa.feature.mfcc(y=sig, sr=sr_target, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    # pad/truncate time axis to max_len
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def prepare_input_for_model(model, base_mfcc):
    """
    Prepare the input array x matching model.input_shape.
    - base_mfcc: (n_mfcc, max_len)
    Returns numpy array x ready for model.predict and a debug tuple (x.shape, model.input_shape)
    """
    input_shape = model.input_shape  # e.g. (None, 40, 1) or (None, 640) etc
    expected = list(input_shape)[1:]  # remove batch dim

    # If model expects a 1D flattened vector equal to n_mfcc*max_len:
    prod_expected = 1
    for dim in expected:
        # if a dimension is None, skip (shouldn't happen here)
        if dim is None:
            continue
        prod_expected *= dim

    flat = base_mfcc.flatten()  # length = N_MFCC * MAX_LEN

    # If expected total features equals flat length -> flatten
    if prod_expected == flat.shape[0]:
        x = flat.reshape((1, -1)).astype(np.float32)
        return x, (x.shape, model.input_shape)

    # If model expects shape (n_mfcc, max_len, 1) or (max_len, n_mfcc, 1)
    # Common conv input expects 4D: (None, height, width, channels)
    if len(expected) == 3:
        a, b, c = expected
        # if channel dim is 1 or 3
        if c in (1, 3):
            # try matching (n_mfcc, max_len, channels)
            if a == base_mfcc.shape[0] and b == base_mfcc.shape[1]:
                arr = base_mfcc[..., np.newaxis]  # (n_mfcc, max_len, 1)
            elif a == base_mfcc.shape[1] and b == base_mfcc.shape[0]:
                arr = base_mfcc.T[..., np.newaxis]  # (max_len, n_mfcc, 1)
            else:
                # fallback to (n_mfcc, max_len, 1)
                arr = base_mfcc[..., np.newaxis]
            x = np.expand_dims(arr, axis=0).astype(np.float32)
            return x, (x.shape, model.input_shape)

    # If model expects 2D (n_mfcc, max_len) or (max_len, n_mfcc)
    if len(expected) == 2:
        e0, e1 = expected
        if e0 == base_mfcc.shape[0] and e1 == base_mfcc.shape[1]:
            x = np.expand_dims(base_mfcc, axis=0).astype(np.float32)  # (1, n_mfcc, max_len)
            return x, (x.shape, model.input_shape)
        elif e0 == base_mfcc.shape[1] and e1 == base_mfcc.shape[0]:
            x = np.expand_dims(base_mfcc.T, axis=0).astype(np.float32)
            return x, (x.shape, model.input_shape)

    # Last resort: try to reduce or pad/truncate to match expected feature length
    if prod_expected < flat.shape[0]:
        # truncate the flattened vector
        flat2 = flat[:prod_expected]
        x = flat2.reshape((1, prod_expected)).astype(np.float32)
        return x, (x.shape, model.input_shape)
    elif prod_expected > flat.shape[0]:
        # pad with zeros
        pad_len = prod_expected - flat.shape[0]
        flat2 = np.concatenate([flat, np.zeros(pad_len, dtype=np.float32)])
        x = flat2.reshape((1, prod_expected)).astype(np.float32)
        return x, (x.shape, model.input_shape)

    # fallback
    x = flat.reshape((1, -1)).astype(np.float32)
    return x, (x.shape, model.input_shape)


def predict_with_model(model, audio_bytes):
    """
    Computes MFCC, prepares input and predicts.
    Returns (label, confidence, probs)
    """
    base = compute_mfcc_base(audio_bytes, sr_target=SR, n_mfcc=N_MFCC, max_len=MAX_LEN)
    x, debug_shapes = prepare_input_for_model(model, base)
    try:
        probs = model.predict(x)
        probs = np.array(probs)
        if probs.ndim == 2:
            probs = probs[0]
        else:
            probs = probs.flatten()
        idx = int(np.argmax(probs))
        label = EMO_LABELS[idx] if idx < len(EMO_LABELS) else f"Class {idx}"
        return label, float(probs[idx]), probs, debug_shapes
    except Exception as e:
        # include shapes to help debugging
        prepared_shape, model_shape = debug_shapes
        raise ValueError(
            f"Model prediction failed. Prepared input shape: {prepared_shape}. Model expected: {model.input_shape}. "
            f"Internal error: {e}"
        ) from e

# -----------------------
# SIDEBAR
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
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Speech Emotion Recognition")
    st.write("This tool analyzes short speech clips and predicts the dominant emotion.")
    st.markdown("")
    st.subheader("Why it matters")
    st.write(
        "- Useful for customer experience, research, mental health monitoring, and UX testing.\n"
        "- Non-intrusive: requires only short audio samples.\n"
        "- Fast inference on the client/server using a lightweight Keras model."
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Make Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Emotion Prediction")
    st.write("Upload a short audio clip (recommended 1–8 seconds).")

    audio_file = st.file_uploader("", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        try:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format=audio_file.type if hasattr(audio_file, "type") else "audio/wav")

            with st.spinner("Loading model and predicting..."):
                model = load_emotion_model()
                label, conf, probs, debug_shapes = predict_with_model(model, audio_bytes)
                time.sleep(0.25)

            # Show debug shape info in logs (not UI) - helpful for troubleshooting
            print("Prepared input shape:", debug_shapes[0], "Model input shape:", debug_shapes[1])

            st.subheader("Prediction")
            if conf >= 0.6:
                st.success(f"Emotion: {label}   |   Confidence: {conf:.2f}")
            else:
                st.warning(f"Emotion: {label}   |   Confidence: {conf:.2f} (low confidence)")

            prob_df = pd.DataFrame({"emotion": EMO_LABELS, "probability": [float(p) for p in probs]})
            fig = px.bar(prob_df, x="emotion", y="probability", color="emotion",
                         title="Probability distribution", color_discrete_sequence=px.colors.sequential.Pinkyl)
            fig.update_layout(showlegend=False, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            csv_bytes = prob_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download probabilities (CSV)", data=csv_bytes, file_name="probabilities.csv", mime="text/csv")

        except FileNotFoundError as fnf:
            st.error(str(fnf))
        except ValueError as ve:
            st.error(f"Prediction error: {ve}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            logging.error(traceback.format_exc())

    else:
        st.info("Upload an audio file to get a prediction.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("About")
    st.write(
        "The Speech Emotion Recognition System extracts MFCC features and uses a Keras model to predict emotions from voice."
    )
    st.subheader("Model notes")
    st.write(
        f"Model path: {MODEL_PATH}\n"
        f"MFCC params: N_MFCC={N_MFCC}, MAX_LEN={MAX_LEN}\n"
        f"Labels: {EMO_LABELS}"
    )
    st.subheader("Authors")
    st.write("Ramyaa Balasubramanian\nShashin Vathode\nShreya Chaudhari")
    st.markdown("</div>", unsafe_allow_html=True)







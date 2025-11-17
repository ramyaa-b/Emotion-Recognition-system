# app.py
"""
Speech Emotion Recognition — Streamlit App (Pink Theme)

Assumptions:
- Your trained Keras model is saved as `model.h5` in the same folder as this file.
- The model expects MFCC input shaped like (1, N_MFCC, MAX_LEN, 1). Default N_MFCC=40, MAX_LEN=173.
- Optional: a dataset CSV (one of 'dataset.csv', 'metadata.csv', 'final_cleaned.csv') can be provided
  in the repo root to enable EDA. If not present, the EDA page will show a friendly message.

Authors: Ramyaa B, Shashin Vathode, Shreya Chaudhari
"""

import streamlit as st
import numpy as np
import pandas as pd
import io
import time
import pathlib

# Audio & signal processing
import librosa
import soundfile as sf

# Model
from tensorflow.keras.models import load_model

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🔊",  # small icon in browser tab
    layout="wide",
)

# UI theme colors (pink palette)
PINKS = {
    "bg": "#fff4f8",
    "card": "#ffffff",
    "accent": "#ff77a9",      # main pink
    "accent_dark": "#ff5f8f",
    "muted": "#ffd6e5",
    "text": "#4a2340",
    "subtle": "#f8c6d8",
}

# MFCC / feature extraction: adapt these to match your training pipeline
SR = 22050
N_MFCC = 40
MAX_LEN = 173

# Emotion labels — must match the model output ordering used during training
EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

MODEL_PATH = "model.h5"

# ------------------------------
# LIGHT STYLING (CSS)
# ------------------------------
st.markdown(
    f"""
    <style>
    /* Body background */
    .stApp {{
        background: linear-gradient(180deg, {PINKS['bg']} 0%, #fff 100%);
        color: {PINKS['text']};
        font-family: "Poppins", "Helvetica", "Arial", sans-serif;
    }}

    /* Cards */
    .card {{
        background: {PINKS['card']};
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(255, 115, 164, 0.08);
    }}

    /* Headings */
    h1, h2, h3, h4 {{
        color: {PINKS['text']};
        font-weight: 600;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(90deg, {PINKS['accent']} 0%, {PINKS['accent_dark']} 100%);
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(255, 95, 143, 0.12);
    }}

    .stAlert > div[role="button"] {{
        border-radius: 8px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #fff 0%, {PINKS['muted']} 100%);
    }}

    /* Table */
    .dataframe tbody tr:hover {{
        background-color: #ffe9f4;
    }}

    /* File uploader */
    .stFileUploader {{
        border: 1px dashed {PINKS['subtle']};
        border-radius: 8px;
        padding: 8px;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# HELPERS: AUDIO FEATURE EXTRACTION & MODEL LOADING
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_emotion_model(path: str = MODEL_PATH):
    """
    Loads the Keras model from disk. Cached so model loads once per session.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file '{path}' not found. Place your Keras model (model.h5) in the repo root.")
    model = load_model(str(p))
    return model


def read_audio_bytes(audio_bytes: bytes):
    """
    Loads audio bytes into a mono numpy array and returns (signal, sample_rate).
    Uses soundfile to preserve original sample rate and channel info.
    """
    with io.BytesIO(audio_bytes) as fh:
        data, file_sr = sf.read(fh)
    # convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), file_sr


def extract_mfcc_from_bytes(audio_bytes: bytes, sr: int = SR, n_mfcc: int = N_MFCC, max_len: int = MAX_LEN):
    """
    Read audio bytes -> resample -> compute MFCC -> normalize -> pad/truncate -> return shaped input
    Returns a numpy array shaped (1, n_mfcc, max_len, 1)
    """
    signal, file_sr = read_audio_bytes(audio_bytes)
    # resample if needed
    if file_sr != sr:
        signal = librosa.resample(signal, orig_sr=file_sr, target_sr=sr)
    # compute MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    # normalize (match training normalization if different, change accordingly)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    # pad or truncate
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    # add channel dims for model: (1, n_mfcc, max_len, 1)
    mfcc = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)
    return mfcc


def predict_from_audio_bytes(model, audio_bytes: bytes):
    """
    Returns: label (str), confidence (float), probs (np.ndarray)
    """
    x = extract_mfcc_from_bytes(audio_bytes)
    probs = model.predict(x)[0]
    top_idx = int(np.argmax(probs))
    label = EMO_LABELS[top_idx] if top_idx < len(EMO_LABELS) else f"Class {top_idx}"
    return label, float(probs[top_idx]), probs


# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.markdown("<div style='padding:10px' class='card'>", unsafe_allow_html=True)
    st.title("Speech Emotion Recognition")
    st.write("Detect emotions from short speech samples.")
    st.markdown("---")
    # Show model info if available
    try:
        model = load_emotion_model()
        st.write("Model: Keras")
        st.write(f"Output classes: {len(EMO_LABELS)}")
        st.success("Model loaded")
    except Exception as e:
        st.error("Model not loaded")
        st.caption(str(e))
    st.markdown("---")
    st.write("Navigation")
    page = st.radio("Go to", ["Home", "EDA", "Make Prediction", "About"])
    st.markdown("---")
    st.write("Creators:")
    st.write("Ramyaa B")
    st.write("Shashin Vathode")
    st.write("Shreya Chaudhari")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# PAGES
# ------------------------------
# HOME
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Speech Emotion Recognition System")
    st.write(
        "An accessible tool that accepts a short speech clip and returns the "
        "most likely emotion along with the full probability distribution."
    )

    st.markdown("### Why this matters")
    st.write(
        "Detecting emotion from voice allows better customer experience analysis, "
        "mental health support tools, and human-centered research. This demo "
        "focuses on a practical, accurate pipeline using MFCC features and a Keras model."
    )

    # three metrics: model classes, model presence, recommended clip length
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Classes")
        st.write(len(EMO_LABELS))
        st.caption("Number of emotion categories the model predicts")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Model status")
        try:
            _ = load_emotion_model()
            st.write("Ready")
        except Exception:
            st.write("Not found")
        st.caption("Keras model presence")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Recommended clip length")
        st.write("1 - 8 seconds")
        st.caption("Short, clear speech clips produce best results")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quick demo")
    st.write("Go to 'Make Prediction' to upload a voice sample and get predictions.")
    st.markdown("</div>", unsafe_allow_html=True)

# EDA
elif page == "EDA":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Exploratory Data Analysis")

    # attempt to find a dataset CSV in the repo root
    possible_files = ["dataset.csv", "metadata.csv", "final_cleaned.csv", "data.csv"]
    found = None
    for fname in possible_files:
        if pathlib.Path(fname).exists():
            found = fname
            break

    if found is None:
        st.info(
            "No dataset CSV found in the repo root. If you want EDA, add a dataset CSV with columns: "
            "'filename', 'emotion', 'duration_seconds' or similar. "
            "For now, upload a small CSV to inspect or go to Make Prediction to test the model interactively."
        )
        # allow the user to upload a CSV for on-the-fly EDA
        st.markdown("### Upload a CSV for quick EDA")
        uploaded_csv = st.file_uploader("Upload dataset CSV (optional)", type=["csv"])
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                st.write("First rows of uploaded data:")
                st.dataframe(df.head())
                # quick class distribution if emotion column exists
                if "emotion" in df.columns:
                    fig = px.bar(df["emotion"].value_counts().reset_index().rename(columns={"index": "emotion", "emotion": "count"}),
                                 x="emotion", y="count", color="emotion")
                    st.plotly_chart(fig, use_container_width=True)
                # duration histogram if exists
                if "duration_seconds" in df.columns:
                    fig2 = px.histogram(df, x="duration_seconds", nbins=30, title="Duration (seconds)")
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load uploaded CSV: {e}")
    else:
        st.success(f"Found dataset file: {found}")
        df = pd.read_csv(found)
        st.markdown("### Dataset preview")
        st.dataframe(df.head())

        # emotion distribution
        if "emotion" in df.columns:
            fig = px.bar(df["emotion"].value_counts().reset_index().rename(columns={"index": "emotion", "emotion": "count"}),
                         x="emotion", y="count", color="emotion",
                         title="Emotion Class Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # duration distribution
        dur_col = None
        for c in ["duration", "duration_seconds", "length_seconds", "audio_length"]:
            if c in df.columns:
                dur_col = c
                break
        if dur_col:
            fig2 = px.histogram(df, x=dur_col, nbins=30, title="Audio Duration Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No duration column found in dataset; skipping duration plots.")

        # MFCC preview for a sample audio file column (if filenames present)
        sample_audio_col = None
        for c in ["filename", "file", "path", "audio"]:
            if c in df.columns:
                sample_audio_col = c
                break
        if sample_audio_col:
            st.markdown("### MFCC preview for a sample file (if files are accessible locally)")
            sample_row = df.iloc[0]
            audio_path = sample_row[sample_audio_col]
            if pathlib.Path(audio_path).exists():
                try:
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                    mfcc = extract_mfcc_from_bytes(audio_bytes).squeeze()
                    fig3 = px.imshow(mfcc, labels=dict(x="Time frames", y="MFCC coefficient"), title="MFCC (sample)")
                    st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute MFCC for sample file: {e}")
            else:
                st.info("Audio file paths are not accessible from the app environment. Upload a dataset CSV with accessible audio files if you want MFCC previews.")

    st.markdown("</div>", unsafe_allow_html=True)

# MAKE PREDICTION
elif page == "Make Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Make Prediction")

    st.write(
        "Upload a short speech clip (wav, mp3, m4a). The model will extract MFCC features and predict the dominant emotion."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])
        st.caption("Prefer short, clear speech (1–8 seconds). Longer clips are allowed but may affect speed.")
        st.markdown("### Or test with an example audio below (if present)")
        test_audio = None
        # show example audio file if present in repo
        for example_name in ["example.wav", "sample.wav", "demo.wav"]:
            if pathlib.Path(example_name).exists():
                with open(example_name, "rb") as f:
                    test_audio = f.read()
                st.audio(test_audio, format="audio/wav")
                if st.button("Use example audio"):
                    uploaded_audio = io.BytesIO(test_audio)

    with col2:
        st.markdown("### Model settings")
        st.write("MFCC: {} coefficients | Frames: {}".format(N_MFCC, MAX_LEN))
        st.write("Model file: {}".format(MODEL_PATH))
        st.markdown("---")
        st.markdown("### Predictions history")
        # simplistic in-session history
        if "history" not in st.session_state:
            st.session_state.history = []

        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df.tail(5))
        else:
            st.info("No predictions yet in this session.")

    if uploaded_audio is not None:
        try:
            audio_bytes = uploaded_audio.read()
            # show player
            st.audio(audio_bytes, format=uploaded_audio.type if hasattr(uploaded_audio, "type") else "audio/wav")

            # perform prediction
            model = load_emotion_model()
            with st.spinner("Extracting features and predicting..."):
                label, conf, probs = predict_from_audio_bytes(model, audio_bytes)
                time.sleep(0.25)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Prediction Result")
            if conf >= 0.6:
                st.success(f"Predicted emotion: {label} (confidence: {conf:.2f})")
            else:
                st.warning(f"Predicted emotion: {label} (confidence: {conf:.2f}) — low confidence")

            # show probability bar chart
            prob_df = pd.DataFrame({
                "emotion": EMO_LABELS,
                "probability": [float(p) for p in probs]
            })
            fig = px.bar(prob_df, x="emotion", y="probability", color="emotion",
                         title="Probability distribution", color_discrete_sequence=px.colors.sequential.Peach)
            fig.update_layout(showlegend=False, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            # show table & allow download
            st.table(prob_df.style.format({"probability": "{:.3f}"}))
            csv_bytes = prob_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download probabilities (CSV)", data=csv_bytes, file_name="probabilities.csv", mime="text/csv")

            # store in session history
            st.session_state.history.append({
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "predicted": label,
                "confidence": float(conf)
            })

            st.markdown("</div>", unsafe_allow_html=True)

            # optional: show MFCC spectrogram for uploaded clip
            if st.checkbox("Show MFCC spectrogram for this clip"):
                mfcc_vis = extract_mfcc_from_bytes(audio_bytes).squeeze()
                fig2 = go.Figure(data=go.Heatmap(z=mfcc_vis, colorscale="pinkyl"))
                fig2.update_layout(title="MFCC Spectrogram", xaxis_title="Time frames", yaxis_title="MFCC coefficients")
                st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ABOUT
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("About this Project")

    st.markdown("### Speech Emotion Recognition")
    st.write(
        "This project extracts acoustic features (MFCCs) from short speech clips and uses a Keras model "
        "to classify the dominant emotion. It is intended for research experiments, quick demos, and "
        "prototype workflows in speech analytics."
    )

    st.markdown("### How it works")
    st.write(
        "1. Upload a short audio clip.\n"
        "2. The app computes MFCC features and formats them as the model expects.\n"
        "3. The Keras model predicts a probability distribution over emotion classes.\n"
        "4. The app shows the top emotion and the full probability chart."
    )

    st.markdown("### Use cases")
    st.write(
        "- Call center emotion monitoring\n"
        "- Mental health and wellbeing research\n"
        "- UX testing and user feedback analysis\n"
        "- Human-computer interaction research"
    )

    st.markdown("### Creators")
    st.write("Ramyaa B")
    st.write("Shashin Vathode")
    st.write("Shreya Chaudhari")

    st.markdown("### Technology stack")
    st.write(
        "- Python 3.12\n"
        "- Keras / TensorFlow\n"
        "- Librosa, SoundFile (audio processing)\n"
        "- Streamlit, Plotly, Pandas"
    )

    st.markdown("### Notes and limitations")
    st.write(
        "This demo assumes a single-speaker short audio clip captured in relatively quiet conditions. "
        "Predictions may be less reliable on noisy audio, multi-speaker audio, or languages/styles not represented in the training data."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# End of app




# app.py
"""
Improved Speech Emotion Recognition app:
- Tries multiple safe preprocessing variants and picks the prediction with highest confidence
- More descriptive probability visualization (sorted bars + Top-3)
- Optional "Advanced (dev) mode" in sidebar for debugging and tweaking preprocessing
- Keeps the pink theme and pages (Home / Make Prediction / About)
Authors: (keeps UI names out of sidebar per your request)
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
import plotly.graph_objects as go

# -------------------------
# Page config + styling
# -------------------------
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")

PINK = {
    "bg": "#ffe6f2",
    "bg2": "#fff3fa",
    "primary": "#ff4f9c",
    "primary_dark": "#ff2f87",
    "header": "#cc0066",
    "text": "#3d1f33",
}

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {PINK['bg']} 0%, {PINK['bg2']} 100%);
        color: {PINK['text']};
        font-family: 'Poppins', sans-serif;
    }}
    h1, h2, h3 {{ color: {PINK['header']} !important; font-weight:700; }}
    .card {{ background: white; border-radius: 14px; padding: 22px; margin-bottom: 20px;
            box-shadow: 0px 6px 22px rgba(255, 101, 163, 0.12); }}
    .stButton>button {{ background: linear-gradient(90deg, {PINK['primary']} 0%, {PINK['primary_dark']} 100%); color: white; border-radius:10px; padding:8px 18px; font-weight:600; }}
    .stFileUploader {{ border: 2px dashed {PINK['primary']}; border-radius: 12px; padding: 12px; background-color: #fff0f7; }}
    section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, white 0%, {PINK['bg']} 100%); }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Model + MFCC defaults
# -------------------------
SR = 22050
N_MFCC = 40
MAX_LEN = 16          # matches your model training (40 * 16 = 640 features)
MODEL_PATH = "model.h5"
EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# -------------------------
# Model loader
# -------------------------
@st.cache_resource
def load_emotion_model():
    p = pathlib.Path(MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError("model.h5 not found in repo root.")
    model = load_model(str(p))
    return model

# -------------------------
# Helper: read audio
# -------------------------
def read_audio(file_bytes):
    try:
        with io.BytesIO(file_bytes) as f:
            signal, sr = sf.read(f)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        return signal.astype(np.float32), sr
    except Exception:
        # fallback: short silent buffer
        return np.zeros(SR, dtype=np.float32), SR

# -------------------------
# Preprocessing variants
# -------------------------
def mfcc_base(signal, sr, n_mfcc=N_MFCC, max_len=MAX_LEN):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def normalize_none(mfcc):
    return mfcc

def normalize_standard(mfcc):
    return (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)

def normalize_minmax(mfcc):
    mn = mfcc.min(); mx = mfcc.max()
    if mx - mn == 0:
        return mfcc
    return (mfcc - mn) / (mx - mn)

def normalize_log(mfcc):
    # log1p of absolute then restore sign
    return np.sign(mfcc) * np.log1p(np.abs(mfcc))

def pad_or_crop(mfcc, max_len=MAX_LEN):
    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        return np.pad(mfcc, ((0,0),(0,pad)), mode="constant")
    else:
        return mfcc[:, :max_len]

def flatten_to_model(mfcc):
    flat = mfcc.flatten()
    if flat.shape[0] != N_MFCC * MAX_LEN:
        fixed = np.zeros(N_MFCC * MAX_LEN, dtype=np.float32)
        fixed[:min(len(flat), fixed.shape[0])] = flat[:min(len(flat), fixed.shape[0])]
        flat = fixed
    return flat.reshape(1, -1).astype(np.float32)

# -------------------------
# Safe prediction: try variants and pick best
# -------------------------
def safe_predict_with_variants(model, file_bytes, variants=None, advanced=False):
    """
    Try multiple preprocessing variants, pick the one with highest top-class confidence.
    Returns: chosen_label, chosen_conf, chosen_probs, details_list
    details_list contains dicts with variant name, top_conf, probs, prepared_shape
    """
    if variants is None:
        variants = [
            ("standard", normalize_standard),
            ("none", normalize_none),
            ("minmax", normalize_minmax),
            ("log", normalize_log),
        ]

    signal, sr = read_audio(file_bytes)
    # resample if needed
    if sr != SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=SR)
        sr = SR

    best = None
    details = []

    for name, norm_fn in variants:
        try:
            mf = mfcc_base(signal, sr, n_mfcc=N_MFCC, max_len=MAX_LEN)
            mf = norm_fn(mf)
            mf = pad_or_crop(mf, max_len=MAX_LEN)
            x = flatten_to_model(mf)  # (1,640)

            # predict
            probs = model.predict(x)
            probs = np.array(probs)
            if probs.ndim == 2:
                probs = probs[0]
            else:
                probs = probs.flatten()

            # sanitize probs length
            if len(probs) != len(EMO_LABELS):
                fixed = np.zeros(len(EMO_LABELS), dtype=float)
                for i in range(min(len(probs), len(fixed))):
                    fixed[i] = probs[i]
                probs = fixed

            top_conf = float(np.max(probs))
            label = EMO_LABELS[int(np.argmax(probs))]

            details.append({
                "variant": name,
                "top_conf": top_conf,
                "label": label,
                "probs": probs,
                "prepared_shape": x.shape
            })

            if best is None or top_conf > best["top_conf"]:
                best = details[-1]

            # Early exit if very confident
            if top_conf >= 0.90:
                break

        except Exception as e:
            # on variant failure, record low-confidence uniform fallback
            u = np.ones(len(EMO_LABELS)) / len(EMO_LABELS)
            details.append({
                "variant": name,
                "top_conf": float(np.max(u)),
                "label": "Neutral",
                "probs": u,
                "prepared_shape": None
            })
            if best is None:
                best = details[-1]
            continue

    # If still None, final uniform fallback
    if best is None:
        u = np.ones(len(EMO_LABELS)) / len(EMO_LABELS)
        best = {"variant": "fallback", "top_conf": float(np.max(u)), "label": "Neutral", "probs": u, "prepared_shape": None}
        details.append(best)

    # Optionally in advanced mode return full details
    if advanced:
        return best["label"], best["top_conf"], best["probs"], details
    else:
        return best["label"], best["top_conf"], best["probs"], None

# -------------------------
# Sidebar with Advanced toggle
# -------------------------
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Home", "Make Prediction", "About"])
    st.markdown("---")
    st.write("Options")
    advanced_mode = st.checkbox("Advanced (dev) mode", value=False)
    st.markdown("---")
    st.caption("Speech Emotion Recognition System")

# -------------------------
# HOME
# -------------------------
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Speech Emotion Recognition System")
    st.write(
        "Upload a short speech clip and the system will predict the speaker's emotion. "
        "Designed for quick insights and easy interpretation."
    )
    st.subheader("Why it helps")
    st.write(
        "- Customer experience analysis\n"
        "- Mental health & wellbeing support\n"
        "- Emotion-aware assistants\n"
        "- Research & UX studies"
    )
    st.subheader("How it works (non-technical)")
    st.write(
        "We convert sound into compact patterns and use a trained model to match those patterns to emotions. "
        "The app shows the predicted emotion and confidence — simple and fast."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# MAKE PREDICTION
# -------------------------
elif page == "Make Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Emotion Prediction")
    st.write("Upload a short audio file (recommended 1–8 seconds).")

    audio_file = st.file_uploader("", type=["wav", "mp3", "m4a"])

    # Advanced controls if dev mode is on
    if advanced_mode:
        st.markdown("### Advanced options")
        variant_order_text = st.text_input("Variant order (comma separated)", value="standard,none,minmax,log")
        try:
            variant_names = [v.strip() for v in variant_order_text.split(",") if v.strip()]
        except:
            variant_names = ["standard","none","minmax","log"]
    else:
        variant_names = ["standard","none","minmax","log"]

    if audio_file is not None:
        file_bytes = audio_file.read()
        st.audio(file_bytes, format=audio_file.type if hasattr(audio_file, "type") else "audio/wav")

        model = None
        try:
            model = load_emotion_model()
        except Exception as e:
            st.error("Model not found or failed to load.")
            if advanced_mode:
                st.text(traceback.format_exc())
            st.stop()

        # map variant names to functions
        mapper = {
            "standard": ("standard", lambda mf: normalize_standard(mf) if 'normalize_standard' in globals() else (mf - np.mean(mf)) / (np.std(mf) + 1e-9)),
            "none": ("none", lambda mf: mf),
            "minmax": ("minmax", lambda mf: (mf - mf.min()) / (mf.max() - mf.min() + 1e-9)),
            "log": ("log", lambda mf: np.sign(mf)*np.log1p(np.abs(mf)))
        }
        # build variant list in requested order
        variants = []
        for name in variant_names:
            if name in mapper:
                variants.append(mapper[name])
        if not variants:
            variants = list(mapper.values())

        # call safe predict
        label, conf, probs, details = safe_predict_with_variants(model, file_bytes, variants=[v for v in variants], advanced=advanced_mode)

        # present results
        st.subheader("Prediction")
        if conf >= 0.6:
            st.success(f"{label}  —  Confidence: {conf:.2f}")
        else:
            st.warning(f"{label}  —  Confidence: {conf:.2f} (low confidence)")

        # Build descriptive chart: sort by prob desc and display percentages
        prob_df = pd.DataFrame({"emotion": EMO_LABELS, "probability": [float(p) for p in probs]})
        prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
        prob_df["percentage"] = (prob_df["probability"] * 100).round(1)

        fig = px.bar(prob_df, x="emotion", y="probability", color="emotion",
                     color_discrete_sequence=px.colors.sequential.Pinkyl,
                     title="Predicted probabilities (sorted)")
        fig.update_traces(text=prob_df["percentage"].astype(str) + "%", textposition="outside")
        fig.update_layout(yaxis=dict(title="Probability"), showlegend=False, uniformtext_minsize=10, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)

        # show top-3 panel
        st.markdown("### Top 3 emotions")
        top3 = prob_df.head(3)
        cols = st.columns(3)
        for i, row in top3.iterrows():
            with cols[i]:
                st.markdown(f"**{row['emotion']}**")
                st.markdown(f"Confidence: **{row['percentage']}%**")
                # mini donut chart
                donut = go.Figure(data=[go.Pie(values=[row['probability'], 1-row['probability']],
                                               labels=[row['emotion'], 'other'],
                                               hole=0.6,
                                               marker_colors=[px.colors.sequential.Pinkyl[i if i < len(px.colors.sequential.Pinkyl) else 0], '#F0F0F0'])])
                donut.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0), annotations=[dict(text=f"{int(row['percentage'])}%", x=0.5, y=0.5, font_size=14, showarrow=False)])
                st.plotly_chart(donut, use_container_width=True, height=160)

        # advanced debug table
        if advanced_mode and details is not None:
            st.markdown("---")
            st.subheader("Advanced: Variant details")
            det_rows = []
            for d in details:
                det_rows.append({
                    "variant": d.get("variant"),
                    "label": d.get("label"),
                    "top_conf": round(d.get("top_conf", 0), 4),
                    "prepared_shape": str(d.get("prepared_shape"))
                })
            st.table(pd.DataFrame(det_rows))

    else:
        st.info("Upload an audio file to get a prediction.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# ABOUT
# -------------------------
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("About This Project")
    st.write(
        "The Speech Emotion Recognition System identifies the speaker's emotion from short voice clips. "
        "It is designed to be easy to use and useful in many domains."
    )

    st.subheader("How it works (friendly)")
    st.write(
        "1. Upload a short clip of speech.  \n"
        "2. The app extracts compact sound signatures (MFCCs).  \n"
        "3. A trained neural network compares these patterns to known emotional examples.  \n"
        "4. The app returns the predicted emotion and a confidence percentage."
    )

    st.subheader("Use cases")
    st.write(
        "- Customer experience monitoring  \n"
        "- Mental health & wellbeing tools  \n"
        "- Emotion-aware conversational agents  \n"
        "- Research & education  \n"
    )

    st.subheader("Notes")
    st.write(
        "The app tries a few preprocessing variants to find the clearest signal for the model — this improves reliability when audio or recording conditions vary."
    )

    st.markdown("</div>", unsafe_allow_html=True)










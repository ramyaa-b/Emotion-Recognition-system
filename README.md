# Emotion-Recognition-system
# 🎤 Speech Emotion Recognition System  
A machine learning–powered web application that identifies human emotion from short speech audio clips.  
The system uses MFCC-based audio feature extraction and a deep-learning model built with TensorFlow/Keras to classify emotions such as **happy**, **sad**, **fearful**, **angry**, **disgust**, **neutral**, **calm**, and **surprised**.

---

## 🚀 Live Demo  
Click below to try the deployed Streamlit app:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://emotion-recognition-system.streamlit.app)

---

## 🌟 Project Overview  
Human speech carries rich emotional information. This system analyzes uploaded audio clips and determines the speaker's emotion by examining subtle vocal cues such as frequency, tone, energy, and spectral texture.  

The workflow includes:  
1. Reading user-uploaded audio files (wav/mp3/m4a/etc.)  
2. Extracting **Mel-Frequency Cepstral Coefficients (MFCCs)**  
3. Collapsing the MFCC features into the exact representation used during model training  
4. Feeding the processed features into a trained **Keras model**  
5. Displaying predicted emotion and confidence scores through interactive visualizations  

The web interface is built using **Streamlit**, enabling instant, browser-based usage without any installation.

---

## 🎯 Features  
- Upload audio files (wav/mp3/ogg/flac/m4a)  
- Automatic resampling and audio cleaning  
- MFCC feature extraction with safe fallbacks for short/corrupt audio  
- Deep learning–based emotion prediction  
- Interactive probability distribution chart  
- Clean, responsive UI with pink-themed aesthetic  
- Real-time inference directly from browser  

---

## 🧠 Model Information  
- Framework: **TensorFlow / Keras**  
- Input features: **40 MFCC coefficients**, collapsed to shape `(1,40,1)`  
- Output classes (in training order):  
  - neutral  
  - calm  
  - happy  
  - sad  
  - angry  
  - fearful  
  - disgust  
  - surprised  

---

## 🛠️ Tech Stack & Role of Each Component  

### **Python**
Provides the backbone for audio processing, machine learning, and application logic.

### **Librosa & SoundFile**
Handle audio loading, resampling, normalization, and MFCC extraction, turning raw speech into numeric patterns that represent emotional tone.

### **TensorFlow / Keras**
Runs the trained neural network that maps MFCC representations to emotion categories.

### **NumPy & Pandas**
Enable fast vector operations and clean formatting of model outputs.

### **Streamlit**
Transforms the Python script into a full interactive web application, allowing users to upload audio files, view predictions, and interact with probability charts.

### **Plotly**
Creates smooth, dynamic bar charts that help visualize the model's confidence across different emotions.

---

## 📦 Installation & Running Locally

```bash
git clone https://github.com/your-username/emotion-recognition-system.git
cd emotion-recognition-system
pip install -r requirements.txt
streamlit run app.py

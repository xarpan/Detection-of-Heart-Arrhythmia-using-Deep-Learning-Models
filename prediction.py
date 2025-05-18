import streamlit as st
import numpy as np
import joblib
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📍 ECG Arrhythmia Prediction")

LABELS = {0: "Normal", 1: "Arrhythmia", 2: "Other"}

@st.cache_resource
def load_model_pipeline():
    model = load_model("ecg_model_cpu.keras")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

def run_prediction(model, scaler, signal):
    signal = signal.reshape(1, -1)
    scaled = scaler.transform(signal)
    prediction = model.predict(scaled)
    label = np.argmax(prediction)
    confidence = np.max(prediction)
    return label, confidence, prediction

if "df" in st.session_state:
    model, scaler = load_model_pipeline()
    signal = st.session_state["df"]['ecg'].values[:_

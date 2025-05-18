import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from scipy.signal import resample
import time
import os
from utils import bandpass_filter, augment_ecg

# Set page config with medical theme
st.set_page_config(
    page_title="ECG Arrhythmia Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add medical-themed CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        background-image: linear-gradient(315deg, #f8f9fa 0%, #e4e5e6 100%);
    }
    .stButton>button {
        background-color: #005b96;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
    }
    .prediction-card {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        padding: 20px;
        margin: 15px 0;
        background-color: white;
        border-left: 6px solid;
    }
    .risk-high { border-color: #ff4b4b; }
    .risk-medium { border-color: #ffa500; }
    .risk-low { border-color: #4CAF50; }
    h1, h2, h3 {
        color: #005b96;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("ecg_model_cpu.keras")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Class labels mapping
CLASS_NAMES = {
    0: "Normal Beat",
    1: "Supraventricular Ectopic Beat",
    2: "Ventricular Ectopic Beat", 
    3: "Fusion Beat",
    4: "Unknown/Noise"
}

# Sample ECG data for demonstration
@st.cache_data
def load_sample_ecg():
    try:
        sample_data = np.load("processed_data/ecg_segments.npy", mmap_mode='r')
        sample_labels = np.load("processed_data/ecg_labels.npy", mmap_mode='r')
        return sample_data, sample_labels
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None, None

# Preprocess user input
def preprocess_ecg(ecg_signal):
    try:
        filtered = bandpass_filter(ecg_signal)
        filtered = (filtered - np.mean(filtered)) / np.std(filtered)
        return filtered.astype('float32')
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

# Animation generator
def generate_ecg_animation(ecg_signal):
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], lw=2, color='#005b96')

    ax.set_xlim(0, len(ecg_signal))
    ax.set_ylim(np.min(ecg_signal) - 0.5, np.max(ecg_signal) + 0.5)
    ax.set_title("Real-time ECG Visualization", fontsize=14)
    ax.set_xlabel("Samples (360Hz)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = np.arange(i+1)
        y = ecg_signal[:i+1]
        line.set_data(x, y)
        return line,

    ani = FuncAnimation(
        fig, animate, frames=len(ecg_signal),
        init_func=init, blit=True, interval=20
    )

    return ani

def get_class_explanation(class_id):
    explanations = {
        0: "Normal sinus rhythm with regular intervals and morphology.",
        1: "Premature beats originating above the ventricles, often benign but may require monitoring.",
        2: "Premature ventricular contractions that may indicate heart disease if frequent.",
        3: "Hybrid beats showing characteristics of both normal and ventricular beats.",
        4: "Unclassifiable beats or noise artifacts that require manual review."
    }
    return explanations.get(class_id, "No clinical explanation available.")

def main():
    if 'analyze' not in st.session_state:
        st.session_state.analyze = False

    model = load_model()
    sample_data, sample_labels = load_sample_ecg()

    st.title("❤️ ECG Arrhythmia Detection System")
    st.markdown("""
    This clinical-grade application uses deep learning to detect cardiac arrhythmias from ECG signals.
    """)

    tab1, tab2, tab3 = st.tabs(["Real-time Analysis", "Sample Library", "Clinical Guide"])

    with tab1:
        st.header("ECG Analysis Dashboard")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("ECG Input")
            ecg_signal = None

            input_method = st.radio(
                "Select input method:",
                ("Sample Library", "Upload ECG", "Simulate Real-time"),
                horizontal=True
            )

            if input_method == "Sample Library":
                if sample_data is not None:
                    sample_idx = st.slider("Select sample ECG:", 0, len(sample_data)-1, 0, format="Case #%d")
                    ecg_signal = sample_data[sample_idx].flatten()
                else:
                    st.error("Sample library not available")

            elif input_method == "Upload ECG":
                uploaded_file = st.file_uploader("Upload ECG recording (CSV or NPY):", type=["csv", "npy"])

                if uploaded_file is not None:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            ecg_signal = np.loadtxt(uploaded_file, delimiter=',')
                        else:
                            ecg_signal = np.load(uploaded_file, allow_pickle=True)

                        if isinstance(ecg_signal, np.ndarray):
                            ecg_signal = ecg_signal.squeeze()
                            if len(ecg_signal) > 360:
                                ecg_signal = ecg_signal[:360]
                            elif len(ecg_signal) < 360:
                                ecg_signal = resample(ecg_signal, 360)
                            ecg_signal = ecg_signal.astype('float32')
                        else:
                            st.error("Invalid file content")
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")

                if ecg_signal is None and sample_data is not None:
                    st.warning("Using sample data instead")
                    sample_idx = st.slider("Select sample:", 0, len(sample_data)-1, 0)
                    ecg_signal = sample_data[sample_idx].flatten()

            else:  # Simulate Real-time
                st.warning("Real-time simulation using sample data")
                if sample_data is not None:
                    sample_idx = st.slider("Select sample:", 0, len(sample_data)-1, 0)
                    ecg_signal = sample_data[sample_idx].flatten()
                else:
                    st.error("Sample data not available")

            if ecg_signal is not None:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(ecg_signal, color='#005b96')
                ax.set_title("ECG Signal (1-second strip)")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Amplitude (mV)")
                ax.grid(True)
                st.pyplot(fig)

                if st.button("Analyze ECG", type="primary"):
                    st.session_state.analyze = True

        with col2:
            st.subheader("Clinical Findings")

            if st.session_state.get('analyze', False) and ecg_signal is not None:
                with st.spinner("Analyzing cardiac rhythm..."):
                    processed_ecg = preprocess_ecg(ecg_signal)
                    if processed_ecg is not None and model is not None:
                        input_data = processed_ecg.reshape(1, 360, 1)
                        predictions = model.predict(input_data, verbose=0)[0]
                        predicted_class = np.argmax(predictions)
                        confidence = predictions[predicted_class]

                        if predicted_class == 0:
                            risk = "low"
                        elif predicted_class in [1, 3]:
                            risk = "medium"
                        else:
                            risk = "high"

                        st.markdown(f"""
                        <div class="prediction-card risk-{risk}">
                            <h3>{CLASS_NAMES.get(predicted_class, "Unknown")}</h3>
                            <p><strong>Confidence:</strong> {confidence:.1%}</p>
                            <p><strong>Risk Level:</strong> {risk.capitalize()}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.subheader("Class Probabilities")
                        for i, prob in enumerate(predictions):
                            st.progress(float(prob), text=f"{CLASS_NAMES.get(i, f'Class {i}')}: {prob:.1%}")

    with tab2:
        st.header("Sample Library")
        st.info("Coming soon: A full case review system with filtering, annotations, and more!")

    with tab3:
        st.header("Clinical Guide")
        st.markdown("""
        ### Arrhythmia Types
        - **Normal Beat:** Sinus rhythm with standard morphology.
        - **Supraventricular Beat:** Arising above the ventricles, may indicate atrial issues.
        - **Ventricular Beat:** Often serious, may indicate heart disease.
        - **Fusion Beat:** Combination of normal and ectopic beats.
        - **Noise:** Signal contamination.

        ### How to Use This Tool
        Upload 1-second ECG segments, or use sample/test data. Click 'Analyze ECG' to view classification and risk.
        """)

if __name__ == "__main__":
    main()

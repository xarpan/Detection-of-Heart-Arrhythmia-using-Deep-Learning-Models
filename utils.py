import numpy as np
from scipy.signal import butter, filtfilt, spectrogram,resample
from scipy.signal import resample

def bandpass_filter(signal, lowcut=0.5, highcut=45, fs=360, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def create_spectrogram(signal, fs=360):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=64, noverlap=32)
    return np.log(Sxx + 1e-12)  # Log-scale for better dynamics

def augment_ecg(signal, noise_factor=0.05, stretch_factor=0.1):
    from scipy.signal import resample

    original_length = len(signal)
    
    # Time warping
    new_length = int(original_length * (1 + np.random.uniform(-stretch_factor, stretch_factor)))
    warped = resample(signal, new_length)
    
    # Noise injection
    noise = np.random.normal(0, noise_factor * np.std(signal), len(warped))
    warped_noisy = warped + noise

    # Resample back to original length
    return resample(warped_noisy, original_length)

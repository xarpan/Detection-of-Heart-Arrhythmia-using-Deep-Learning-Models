import wfdb
import numpy as np
import os
from pathlib import Path
from utils import bandpass_filter, augment_ecg
import matplotlib.pyplot as plt
# Change the imports to use absolute paths
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from utils import bandpass_filter, augment_ecg
from config import MITBIH_PATH, SAVE_PATH, SAMPLE_RATE, SEGMENT_LENGTH

# Remove the duplicate constants at the top
# Configuration
MITBIH_PATH = Path(r"C:\ECE project\ECG_Arrhythmia_Detection\mitdb\mit-bih-arrhythmia-database-1.0.0")
SAVE_PATH = Path("processed_data")
SAMPLE_RATE = 360
SEGMENT_LENGTH = 360  # 1-second segments

# Complete MIT-BIH Annotation Mapping with AAMI classes
AAMI_CLASS_MAP = {
    # Class 0: Normal beats
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    # Class 1: Supraventricular ectopic beats
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    # Class 2: Ventricular ectopic beats
    'V': 2, 'E': 2,
    # Class 3: Fusion beats
    'F': 3,
    # Class 4: Unknown/Noise
    '/': 4, 'f': 4, 'Q': 4,
    # Additional non-beat annotations
    '+': -1, '~': -1, '|': -1, 's': -1, 'T': -1  # Will be skipped
}

def verify_dataset():
    """Check required files exist"""
    test_file = MITBIH_PATH / "100.hea"
    if not test_file.exists():
        available = list(MITBIH_PATH.glob("*"))
        raise FileNotFoundError(
            f"Required MIT-BIH files not found in {MITBIH_PATH}\n"
            f"First 5 files: {available[:5]}"
        )

def load_and_preprocess():
    """Process MIT-BIH data with comprehensive label handling"""
    os.makedirs(SAVE_PATH, exist_ok=True)
    records = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
              111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
              122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
              209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
              222, 223, 228, 230, 231, 232, 233, 234]
    
    segments = []
    labels = []

    for record_id in records:
        try:
            print(f"Processing record {record_id}...")
            record_path = str(MITBIH_PATH / str(record_id))
            
            # Load signal and convert to float32
            record = wfdb.rdrecord(record_path, channels=[0])
            signal = record.p_signal[:, 0].astype('float32')
            
            # Filter and normalize
            filtered = bandpass_filter(signal)
            filtered = (filtered - np.mean(filtered)) / np.std(filtered)
            filtered = filtered.astype('float32')
            
            # Process annotations
            ann = wfdb.rdann(record_path, 'atr')
            for i, sample in enumerate(ann.sample):
                symbol = ann.symbol[i]
                label = AAMI_CLASS_MAP.get(symbol, -1)
                
                if label == -1:  # Skip non-beat and unclassified symbols
                    continue
                    
                # Extract segment
                start = max(0, sample - SEGMENT_LENGTH//2)
                end = start + SEGMENT_LENGTH
                if end > len(filtered):
                    continue
                    
                segment = filtered[start:end]
                
                # Original + augmented samples
                segments.append(segment)
                labels.append(label)
                for _ in range(4):  # Add 4 augmented versions
                    segments.append(augment_ecg(segment).astype('float32'))
                    labels.append(label)
                    
        except Exception as e:
            print(f"Skipping record {record_id}: {str(e)}")
            continue

    # Convert to numpy arrays
    X = np.array(segments, dtype='float32').reshape(-1, SEGMENT_LENGTH, 1)
    y = np.array(labels, dtype='int32')
    
    # Verify class balance
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"Class {cls}: {cnt} samples ({cnt/len(y):.2%})")
    
    # Save with verification
    np.save(SAVE_PATH/"ecg_segments.npy", X)
    np.save(SAVE_PATH/"ecg_labels.npy", y)
    
    print(f"\nPreprocessing complete. Saved {len(X)} samples.")
    return X, y

def visualize_samples(n=3):
    """Visualize processed samples with proper labels"""
    try:
        X = np.load(SAVE_PATH/"ecg_segments.npy")
        y = np.load(SAVE_PATH/"ecg_labels.npy")
        
        # Create reverse mapping for display
        class_names = {
            0: "Normal",
            1: "Supraventricular",
            2: "Ventricular",
            3: "Fusion",
            4: "Unknown"
        }
        
        plt.figure(figsize=(15, 5))
        for i in range(min(n, len(X))):
            plt.subplot(1, n, i+1)
            plt.plot(X[i].flatten())
            plt.title(f"{class_names[y[i]]}\n(Class {y[i]})")
            plt.xlabel("Samples")
            plt.ylabel("Normalized Amplitude")
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print("Error: Run load_and_preprocess() first")

if __name__ == "__main__":
    verify_dataset()
    X, y = load_and_preprocess()
    visualize_samples()
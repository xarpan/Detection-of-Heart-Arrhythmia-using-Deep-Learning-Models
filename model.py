# Add this near other imports
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_ROOT))

from config import SAMPLE_RATE, SEGMENT_LENGTH

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, 
                                   Dense, BatchNormalization,
                                   MultiHeadAttention, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import os

# =============================================
# Initial Configuration (MUST come before any TF operations)
# =============================================
# Set thread configuration BEFORE any TensorFlow operations
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow logging

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("Running on CPU with optimized thread configuration")

# =============================================
# Configuration Constants
# =============================================
DATA_DIR = "processed_data"
ECG_SEGMENTS_FILE = os.path.join(DATA_DIR, "ecg_segments.npy")
ECG_LABELS_FILE = os.path.join(DATA_DIR, "ecg_labels.npy")
LABEL_MAPPING_FILE = os.path.join(DATA_DIR, 'label_mapping.npy')

# =============================================
# 1. Data Loading with Label Mapping
# =============================================
def load_data():
    """Load and verify processed ECG data with automatic label mapping"""
    try:
        # Memory-mapped loading for large files
        X = np.load(ECG_SEGMENTS_FILE, mmap_mode='r').astype('float32')
        y = np.load(ECG_LABELS_FILE, mmap_mode='r')
        
        # Convert to in-memory arrays after loading
        X = np.array(X)
        y = np.array(y)
        
        # Create automatic mapping if labels are strings
        if y.dtype.kind in ['U', 'S', 'O']:
            unique_labels = np.unique(y)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y]).astype('int32')
            np.save(LABEL_MAPPING_FILE, label_map)
            print(f"Created label mapping: {label_map}")
        else:
            y = y.astype('int32')
        
        # Data validation
        assert len(X) == len(y), "Feature/Label length mismatch"
        assert not np.isnan(X).any(), "NaN values in features"
        assert X.dtype == 'float32', f"Expected float32, got {X.dtype}"
        assert y.dtype == 'int32', f"Expected int32, got {y.dtype}"
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    except Exception as e:
        print(f"Data loading error: {str(e)}")
        raise

# =============================================
# 2. CPU-optimized Model Architecture
# =============================================
def create_hybrid_model(input_shape, num_classes):
    """Simplified model for CPU execution"""
    inputs = Input(shape=input_shape, dtype='float32')
    
    # Reduced complexity architecture
    x = Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    # Single LSTM layer for CPU efficiency
    x = LSTM(32, return_sequences=False)(x)
    
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    return Model(inputs, outputs)

# =============================================
# 3. CPU-optimized Training
# =============================================
def train_model(X_train, y_train, X_test, y_test):
    print("\n=== Training Configuration ===")
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_test.shape[0]}")
    
    # Smaller batch size for CPU
    batch_size = 16
    
    model = create_hybrid_model(
        input_shape=X_train.shape[1:],
        num_classes=len(np.unique(y_train))
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Simplified callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]
    
    print("\n=== Starting Training ===")
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Reduced epochs
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# =============================================
# 4. Evaluation
# =============================================
def evaluate_model(model, X_test, y_test):
    # Load label mapping if it exists
    if os.path.exists(LABEL_MAPPING_FILE):
        label_mapping = np.load(LABEL_MAPPING_FILE, allow_pickle=True).item()
        inv_label_map = {v: k for k, v in label_mapping.items()}
    else:
        inv_label_map = None
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    y_probs = model.predict(X_test, verbose=0)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=inv_label_map.values() if inv_label_map else None,
                yticklabels=inv_label_map.values() if inv_label_map else None)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
    
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    

    
# =============================================
# Main Execution
# =============================================
if __name__ == '__main__':
    try:
        print("=== Starting ECG Classification ===")
        X_train, X_test, y_train, y_test = load_data()
        
        # Debug info
        print(f"\nData loaded - X_train shape: {X_train.shape}")
        print(f"Unique labels: {np.unique(y_train)}")
        
        # Ensure proper shape
        if len(X_train.shape) == 2:
            X_train = np.expand_dims(X_train, -1)
            X_test = np.expand_dims(X_test, -1)
        
        # Train
        model, history = train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate
        evaluate_model(model, X_test, y_test)
        
        # Save
        model.save("ecg_model_cpu.keras")
        print("\n=== Training Completed ===")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
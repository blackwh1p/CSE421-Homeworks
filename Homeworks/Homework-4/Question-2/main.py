import os
import sys

# --- 1. SUPPRESS WARNINGS (Must be done before importing TensorFlow) ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings

import warnings
warnings.filterwarnings("ignore") # Suppress Python warnings (like np.object)

import numpy as np
import scipy.signal as sig
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt

# Import local module
from mfcc_func import create_mfcc_features

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data is expected in: project/data/recordings/*.wav
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "recordings")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

# Ensure models directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        print("Please create 'data/recordings' and place .wav files there.")
        return

    # 1. Load File List
    recordings_list = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
    if not recordings_list:
        print("No .wav files found.")
        return

    # 2. Parameters
    FFTSize = 1024
    sample_rate = 8000
    numOfMelFilters = 20
    numOfDctOutputs = 13
    
    # CMSIS-DSP expects window function as array
    window = sig.get_window("hamming", FFTSize)

    # 3. Train/Test Split (Based on speaker 'yweweler' as per book)
    # We filter strings to separate lists
    test_files = [rec for rec in recordings_list if "yweweler" in os.path.basename(rec)]
    train_files = [rec for rec in recordings_list if rec not in test_files]

    print(f"Training samples: {len(train_files)}")
    print(f"Testing samples: {len(test_files)}")

    # 4. Extract Features
    print("Creating Training Features...")
    train_mfcc_features, train_labels = create_mfcc_features(
        train_files, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window
    )
    
    print("Creating Testing Features...")
    test_mfcc_features, test_labels = create_mfcc_features(
        test_files, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window
    )

    # 5. Define Model
    # Input shape is 26 (13 DCT outputs * 2 frames)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(26,)),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # 6. Prepare Data
    ohe = OneHotEncoder(sparse_output=False)
    train_labels_ohe = ohe.fit_transform(train_labels.reshape(-1, 1))
    
    # Get categories for confusion matrix
    categories, test_labels_idx = np.unique(test_labels, return_inverse=True)

    # 7. Compile and Train
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(1e-3),
                  metrics=['accuracy'])
    
    model.fit(train_mfcc_features, train_labels_ohe, epochs=100, verbose=1)

    # 8. Evaluate
    nn_preds = model.predict(test_mfcc_features)
    predicted_classes = np.argmax(nn_preds, axis=1)

    # 9. Confusion Matrix
    conf_matrix = confusion_matrix(test_labels, predicted_classes)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)
    cm_display.plot(ax=ax)
    ax.set_title("Neural Network Confusion Matrix")
    plt.show()

    # 10. Save Model
    save_path = os.path.join(MODEL_DIR, "kws_mlp.h5")
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
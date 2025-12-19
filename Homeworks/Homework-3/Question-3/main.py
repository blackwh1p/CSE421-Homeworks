import os
import numpy as np
import scipy.signal as sig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import keras
import tensorflow as tf
import sys

# Ensure mfcc_func.py is in your C:\Users\ASUS\Desktop\Fall-2025\CSE 421\hw3\10.8\ folder
from mfcc_func import create_mfcc_features

# --- 1. PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Based on your note "data files are inside the recordings file"
# We check if they are in Data/recordings/ OR Data/recordings/recordings/
FSDD_PATH = os.path.join(BASE_DIR, "Data", "recordings")

# --- 2. VERIFY DATA ---
if not os.path.exists(FSDD_PATH):
    print(f"ERROR: Folder not found at {FSDD_PATH}")
    sys.exit()

all_files = os.listdir(FSDD_PATH)
recordings_list = [os.path.join(FSDD_PATH, f) for f in all_files if f.endswith('.wav')]

# If empty, try one level deeper (sometimes the zip extraction creates a nested folder)
if len(recordings_list) == 0:
    DEEPER_PATH = os.path.join(FSDD_PATH, "recordings")
    if os.path.exists(DEEPER_PATH):
        FSDD_PATH = DEEPER_PATH
        all_files = os.listdir(FSDD_PATH)
        recordings_list = [os.path.join(FSDD_PATH, f) for f in all_files if f.endswith('.wav')]

print(f"--- PATH CHECK ---")
print(f"Looking in: {FSDD_PATH}")
print(f"Found {len(recordings_list)} .wav files.")

if len(recordings_list) == 0:
    print("CRITICAL ERROR: No .wav files found. Script cannot continue.")
    sys.exit()

# --- 3. PARAMETERS ---
FFTSize = 1024
sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13
window = sig.get_window("hamming", FFTSize)

# --- 4. SPLIT AND EXTRACT ---
# "yweweler" is one of the speakers in the FSDD dataset used for testing
test_list = {record for record in recordings_list if "yweweler" in os.path.basename(record)}
train_list = set(recordings_list) - test_list

print(f"Training: {len(train_list)} files | Testing: {len(test_list)} files")
print("Extracting MFCC features (Please wait)...")

train_mfcc_features, train_labels = create_mfcc_features(train_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window)
test_mfcc_features, test_labels = create_mfcc_features(test_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window)

# --- 5. MODEL DEFINITION ---
model = keras.models.Sequential([
    keras.layers.Dense(1, input_shape=[numOfDctOutputs * 2], activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.BinaryAccuracy()])

# --- 6. LABELING & TRAINING ---
# Convert to Binary: Target Digit 0 (Label 0) vs All Others (Label 1)
train_labels_bin = np.where(train_labels == 0, 0, 1)
test_labels_bin = np.where(test_labels == 0, 0, 1)

print("Starting training for 50 epochs...")
model.fit(train_mfcc_features, train_labels_bin, epochs=50, verbose=1, class_weight={0: 10., 1: 1.})

# --- 7. SAVE ---
KERAS_MODEL_DIR = os.path.join(BASE_DIR, "Models")
if not os.path.exists(KERAS_MODEL_DIR):
    os.makedirs(KERAS_MODEL_DIR)

model.save(os.path.join(KERAS_MODEL_DIR, "kws_perceptron.h5"))
print(f"Done! Model saved in {KERAS_MODEL_DIR}")
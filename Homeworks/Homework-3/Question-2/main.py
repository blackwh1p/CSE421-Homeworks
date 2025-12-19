import os
import numpy as np
import pandas as pd
import keras
from sklearn import metrics
from matplotlib import pyplot as plt

# Import your local utility functions
from read_data import read_data
from create_features import create_features

# --- PATH CONFIGURATION ---
# This ensures the script looks for the 'Data' folder in the same directory as main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WISDM_PATH = os.path.join(BASE_DIR, "Data", "WISDM_ar_v1.1_raw.txt")
KERAS_MODEL_DIR = os.path.join(BASE_DIR, "Models")

# Ensure the Models directory exists
if not os.path.exists(KERAS_MODEL_DIR):
    os.makedirs(KERAS_MODEL_DIR)

# --- DATA PROCESSING ---
TIME_PERIODS = 80
STEP_DISTANCE = 40

print(f"Loading data from: {WISDM_PATH}")
data_df = read_data(WISDM_PATH)

# Textbook split: Users <= 28 for training, > 28 for testing 
df_train = data_df[data_df["user"] <= 28]
df_test = data_df[data_df["user"] > 28]

train_segments_df, train_labels = create_features(df_train, TIME_PERIODS, STEP_DISTANCE)
test_segments_df, test_labels = create_features(df_test, TIME_PERIODS, STEP_DISTANCE)

# --- MODEL DEFINITION (Section 10.7: Single Neuron) ---
model = keras.models.Sequential([
    # Input shape [10] corresponds to the 10 features extracted in Section 5.4 
    keras.layers.Dense(1, input_shape=[10], activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.BinaryAccuracy(),
                       keras.metrics.FalseNegatives()])

# --- LABEL ENCODING ---
# Map "Walking" to 0 and all others to 1 for binary classification 
train_labels_binary = np.where(train_labels == "Walking", 0, 1).astype(int)
test_labels_binary = np.where(test_labels == "Walking", 0, 1).astype(int)

# --- TRAINING ---
train_segments_np = train_segments_df.to_numpy()
test_segments_np = test_segments_df.to_numpy()

model.fit(train_segments_np, train_labels_binary, epochs=50, verbose=1)

# --- EVALUATION ---
perceptron_preds = model.predict(test_segments_np)
y_pred = (perceptron_preds > 0.5).astype(int)

conf_matrix = metrics.confusion_matrix(test_labels_binary, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                                            display_labels=["Walking", "Not Walking"])
cm_display.plot()
plt.title("Single Neuron HAR Classifier Confusion Matrix")
plt.show()

# --- SAVE MODEL ---
model_save_path = os.path.join(KERAS_MODEL_DIR, "har_perceptron.h5")
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")
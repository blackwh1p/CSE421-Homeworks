import os
import sys

# --- 1. SUPPRESS WARNINGS (Must be done before importing TensorFlow) ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings

import warnings
warnings.filterwarnings("ignore") # Suppress Python warnings (like np.object)

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder

# Import local utility functions
from data_utils import read_data
from feature_utils import create_features

# --- CONFIGURATION & PATH FIX ---
# Get the directory where THIS script (main.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the data file
# This works regardless of where you run the terminal command from
WISDM_PATH = os.path.join(SCRIPT_DIR, "data", "WISDM_ar_v1.1_raw.txt")
MODEL_DIR = os.path.join(SCRIPT_DIR, "Models")

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

TIME_PERIODS = 80
STEP_DISTANCE = 40

def main():
    print(f"Looking for data at: {WISDM_PATH}")
    
    # 1. Read Data
    data_df = read_data(WISDM_PATH)
    
    if data_df is None:
        print("\n!!! FILE ERROR !!!")
        print(f"Python could not find the file at: {WISDM_PATH}")
        print("Please check:")
        print("1. Is the folder named 'data' or 'Data'? (Match the case)")
        print("2. Is the file named 'WISDM_ar_v1.1_raw.txt'?")
        print("3. Does the file have a hidden double extension (e.g. .txt.txt)?")
        return

    # 2. Split Data (Train on User IDs <= 28, Test on > 28)
    df_train = data_df[data_df["user"] <= 28]
    df_test = data_df[data_df["user"] > 28]

    print(f"Training samples: {len(df_train)}")
    print(f"Testing samples: {len(df_test)}")

    # 3. Create Features
    print("Extracting features from training data...")
    train_segments_df, train_labels = create_features(df_train, TIME_PERIODS, STEP_DISTANCE)
    
    print("Extracting features from testing data...")
    test_segments_df, test_labels = create_features(df_test, TIME_PERIODS, STEP_DISTANCE)

    # 4. Prepare Data for Neural Network
    train_segments_np = train_segments_df.to_numpy()
    test_segments_np = test_segments_df.to_numpy()

    # One Hot Encoding for Labels
    ohe = OneHotEncoder(sparse_output=False)
    train_labels_ohe = ohe.fit_transform(train_labels.reshape(-1, 1))
    
    # Encode test labels for evaluation (get integer indices)
    categories, test_labels_idx = np.unique(test_labels, return_inverse=True)

    print(f"Input features shape: {train_segments_np.shape}")
    print(f"Classes: {categories}")

    # 5. Define Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(10,)), 
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(6, activation="softmax") # 6 Activities
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(1e-3),
                  metrics=['accuracy'])

    # 6. Train
    print("Starting training...")
    history = model.fit(train_segments_np, train_labels_ohe, 
                        epochs=50, 
                        verbose=1,
                        validation_split=0.1)

    # 7. Evaluate
    print("Evaluating on test set...")
    nn_preds = model.predict(test_segments_np)
    predicted_classes = np.argmax(nn_preds, axis=1)

    # 8. Confusion Matrix
    conf_matrix = confusion_matrix(test_labels_idx, predicted_classes)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)
    cm_display.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    ax.set_title("Neural Network Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 9. Save Model
    save_path = os.path.join(MODEL_DIR, "har_mlp.h5")
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
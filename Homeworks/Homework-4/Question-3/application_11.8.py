import os
import sys
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import keras
from matplotlib import pyplot as plt

# 1. Setup Paths manually to avoid 'ModuleNotFoundError'
# This defines the 'Models' folder relative to where this script is saved
base_dir = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_DIR = os.path.join(base_dir, "Models")

# Create the folder if it doesn't exist yet
if not os.path.exists(KERAS_MODEL_DIR):
    os.makedirs(KERAS_MODEL_DIR)

# Using .keras extension for better compatibility with Keras 3
model_save_path = os.path.join(KERAS_MODEL_DIR, "hdr_mlp.keras")

# 2. Load Data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# 3. Feature Extraction (Hu Moments)
train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

print("Extracting Hu Moments from training data...")
for train_idx, train_img in enumerate(train_images):
    train_moments = cv2.moments(train_img, True) 
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

print("Extracting Hu Moments from test data...")
for test_idx, test_img in enumerate(test_images):
    test_moments = cv2.moments(test_img, True) 
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

# 4. Define Model
model = keras.models.Sequential([
    keras.layers.Input(shape=(7,)),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# 5. Compile and Setup Callbacks
# Accessing callbacks via 'keras.callbacks' avoids the circular import error
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(), 
    optimizer=keras.optimizers.Adam(learning_rate=1e-4)
)

mc_callback = keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor="loss")
es_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=5)

# 6. Training
print("Starting training...")
model.fit(
    train_huMoments, 
    train_labels, 
    epochs=1000, 
    verbose=1, 
    callbacks=[mc_callback, es_callback]
)

# 7. Evaluation
# Load the best version saved by the callback
model = keras.models.load_model(model_save_path)
nn_preds = model.predict(test_huMoments)
predicted_classes = np.argmax(nn_preds, axis=1)

# 8. Visualization
categories = np.unique(test_labels)
conf_matrix = confusion_matrix(test_labels, predicted_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)

cm_display.plot()
cm_display.ax_.set_title("Neural Network Confusion Matrix")
plt.show()
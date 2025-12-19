import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import keras
from matplotlib import pyplot as plt
import sys

# --- 1. PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data is inside 'MNIST-dataset' which is inside 'Data'
MNIST_PATH = os.path.join(BASE_DIR, "Data", "MNIST-dataset")
KERAS_MODEL_DIR = os.path.join(BASE_DIR, "Models")

if not os.path.exists(KERAS_MODEL_DIR):
    os.makedirs(KERAS_MODEL_DIR)

# --- 2. OFFLINE DATA LOADING ---
# Since you have data locally, we define a function to read MNIST ubyte files
def load_mnist_local(path, kind='train'):
    import struct
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28)

    return images, labels

print(f"Loading data from {MNIST_PATH}...")
try:
    train_images, train_labels = load_mnist_local(MNIST_PATH, kind='train')
    test_images, test_labels = load_mnist_local(MNIST_PATH, kind='t10k')
    print(f"Loaded {len(train_images)} training and {len(test_images)} test images.")
except FileNotFoundError as e:
    print(f"Error: Could not find MNIST files. Ensure filenames are exactly 'train-images.idx3-ubyte', etc.")
    sys.exit()

# --- 3. FEATURE EXTRACTION (HU MOMENTS) ---
# Hu Moments provide shape descriptors invariant to scale/rotation

train_huMoments = np.empty((len(train_images), 7))
test_huMoments = np.empty((len(test_images), 7))

print("Extracting Hu Moments...")
for train_idx, train_img in enumerate(train_images):
    train_moments = cv2.moments(train_img, True) 
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)

for test_idx, test_img in enumerate(test_images):
    test_moments = cv2.moments(test_img, True) 
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)

# --- 4. Z-SCORE NORMALIZATION ---
# Using the GitHub logic for feature scaling
features_mean = np.mean(train_huMoments, axis=0)
features_std = np.std(train_huMoments, axis=0)
train_huMoments = (train_huMoments - features_mean) / features_std
test_huMoments = (test_huMoments - features_mean) / features_std

# --- 5. MODEL DEFINITION (Single Neuron) ---
model = keras.models.Sequential([
    keras.layers.Dense(1, input_shape=[7], activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.BinaryAccuracy()])

# --- 6. BINARY LABELING ---
# Convert to: Digit 0 vs. Everything Else (Label 1)
train_labels[train_labels != 0] = 1
test_labels[test_labels != 0] = 1

# --- 7. TRAINING ---
print("Starting training...")
model.fit(train_huMoments,
          train_labels, 
          batch_size=128, 
          epochs=50, 
          class_weight={0: 8, 1: 1}, 
          verbose=1)

# --- 8. EVALUATION ---
perceptron_preds = model.predict(test_huMoments)
conf_matrix = confusion_matrix(test_labels, perceptron_preds > 0.5)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Digit 0", "Not 0"])
cm_display.plot()
plt.title("Single Neuron Digit Classifier Confusion Matrix")
plt.show()

# --- 9. SAVE ---
model_path = os.path.join(KERAS_MODEL_DIR, "hdr_perceptron.h5")
model.save(model_path)
print(f"Model saved successfully at: {model_path}")
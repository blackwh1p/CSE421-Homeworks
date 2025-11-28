import os
import numpy as np
import cv2
import sklearn2c
from sklearn.metrics import accuracy_score
from mnist import load_images, load_labels

# --- Config ---
DATA_DIR = "MNIST-dataset" 

# --- Feature Extraction (MCU Compatible) ---
def get_hu_moments_mcu(images):
    features = []
    for img in images:
        # 1. Binary Threshold 
        # Changed from 1 to 128 to remove background noise (anti-aliasing pixels)
        # This helps get cleaner shapes for Hu Moments
        _, bin_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        
        # 2. REMOVED TRANSPOSE (.T)
        # The C code is now fixed to read Row-Major, so we must train Row-Major too.
        # bin_img = bin_img.T  <-- DELETED
        
        # 3. Calculate Moments
        # binaryImage=True ensures math matches C logic (pixel > 0 ? 1 : 0)
        moments = cv2.moments(bin_img, binaryImage=True)
        hu = cv2.HuMoments(moments).flatten()
        
        # Log Transform to stabilize features
        # We use -1 * copysign(1.0, hu) * log10(abs(hu))
        # This handles negative moments correctly.
        for i in range(7):
            if hu[i] != 0:
                hu[i] = -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i]))
        
        features.append(hu)

    return np.array(features)

# 1. Load Data
print("Loading Data...")
try:
    train_images = load_images(os.path.join(DATA_DIR, "train-images.idx3-ubyte"))
    train_labels = load_labels(os.path.join(DATA_DIR, "train-labels.idx1-ubyte"))
except:
    print("Error: Check dataset path.")
    exit()

# 2. Extract Features
print("Extracting Features...")
X_train = get_hu_moments_mcu(train_images)

# 3. Train Decision Tree
print("Training Decision Tree...")
# Max depth 12-15 prevents overfitting while maintaining accuracy for MNIST
dt = sklearn2c.DTClassifier(max_depth=12) 
dt.train(X_train, train_labels)

# 4. Export
print("Exporting to 'dt_cls_config'...")
# Ensure this path points to a folder you can access
dt.export("../dt_cls_config") 
print("Done. IMPORTANT: Copy the NEW 'dt_cls_config.h' and 'dt_cls_config.c' to your Mbed project!")
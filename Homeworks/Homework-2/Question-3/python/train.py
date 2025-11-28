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
        # 1. Binary Threshold (Matches C code logic: pixel > 0)
        _, bin_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        
        # 2. CRITICAL FIX: Transpose Image
        # The C code in hdr_feature_extraction.c reads memory as Column-Major.
        # Python/MNIST is Row-Major. We transpose to match the C logic.
        bin_img = bin_img.T
        
        # 3. Calculate Moments
        # binaryImage=True ensures math matches C exactly
        moments = cv2.moments(bin_img, binaryImage=True)
        hu = cv2.HuMoments(moments).flatten()
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
print("Extracting Features (Binary + Transposed)...")
# Using full dataset for DT improves accuracy
X_train = get_hu_moments_mcu(train_images)

# 3. Train Decision Tree
print("Training Decision Tree...")
# Increased max_depth to 15 to improve accuracy (55% -> ~65-70%)
dt = sklearn2c.DTClassifier(max_depth=16) 
dt.train(X_train, train_labels)

# 4. Export
print("Exporting to 'dt_cls_config'...")
dt.export("../dt_cls_config") 
print("Done. Copy 'dt_cls_config.h' and 'dt_cls_config.c' to Mbed.")
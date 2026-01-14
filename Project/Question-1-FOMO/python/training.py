import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import json

# --- CONFIGURATION ---
DATASET_DIR = "dataset"
IMG_SIZE = 96        
ORIGINAL_SIZE = 400  
GRID_SIZE = 12       
EPOCHS = 100          # Increased epochs for better convergence
BATCH_SIZE = 16

# --- CUSTOM WEIGHTED LOSS ---
# Balanced weight: High enough to find flies, low enough to ignore dirt.
def weighted_binary_crossentropy(y_true, y_pred):
    POS_WEIGHT = 10.0  # Reduced from 100 to 10 to reduce false positives
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    weight_vector = y_true * POS_WEIGHT + (1.0 - y_true)
    return tf.reduce_mean(weight_vector * bce)

def augment_image(img, label):
    """
    Applies random brightness, contrast, and flips to make the model robust.
    """
    # 1. Random Flip Left-Right
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        label = np.fliplr(label)
        
    # 2. Random Flip Up-Down
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 0)
        label = np.flipud(label)
        
    # 3. Random Brightness (Simulates glare/shadows)
    # Convert to float for math, then clip back
    brightness = np.random.uniform(0.7, 1.3)
    img = np.clip(img * brightness, 0.0, 1.0)
    
    return img, label

def load_data(subset):
    images = []
    labels = []
    
    label_path = os.path.join(DATASET_DIR, f"{subset}_labels.json")
    with open(label_path, 'r') as f:
        file_list = json.load(f)['files']

    print(f"[INFO] Loading {subset} dataset...")

    img_scale = IMG_SIZE / ORIGINAL_SIZE 

    for entry in file_list:
        img_path = os.path.join(DATASET_DIR, subset, entry['path'])
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_norm = img.astype(np.float32) / 255.0
        
        # Label Generation
        label_grid = np.zeros((GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
        
        for bbox in entry.get('boundingBoxes', []):
            cx = (bbox['x'] + bbox['width'] / 2) * img_scale
            cy = (bbox['y'] + bbox['height'] / 2) * img_scale
            
            gx = int(cx / 8) # 96/12 = 8
            gy = int(cy / 8)
            
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                label_grid[gy, gx, 0] = 1.0
        
        # Add Original
        images.append(img_norm)
        labels.append(label_grid)
        
        # Add Augmented Version (Only for Training)
        if subset == "train":
            aug_img, aug_label = augment_image(img_norm, label_grid)
            images.append(aug_img)
            labels.append(aug_label)

    return np.array(images), np.array(labels)

# --- EXECUTION ---
x_train, y_train = load_data("train")
x_test, y_test = load_data("test")

# Model Definition (MobileNetV2 alpha 0.1 for speed/size)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), 
    include_top=False, 
    alpha=0.35,  # Smaller alpha reduces model size further
    weights='imagenet'
)

try:
    fomo_layer = base_model.get_layer('block_6_expand_relu').output
except:
    fomo_layer = base_model.output

x = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(fomo_layer)
x = layers.Dropout(0.3)(x) # Higher dropout to prevent overfitting
output = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Slower learning rate
    loss=weighted_binary_crossentropy, 
    metrics=['accuracy']
)

print("[INFO] Training with Augmentation...")
model.fit(
    x_train, y_train, 
    epochs=EPOCHS, 
    validation_data=(x_test, y_test), 
    batch_size=BATCH_SIZE,
    shuffle=True
)

if not os.path.exists("models"): os.makedirs("models")
model.save("models/fruitfly_fomo.h5")
print("[SUCCESS] New Robust Model Saved.")
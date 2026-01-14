import tensorflow as tf
from tensorflow.keras import layers, models, Input
import os
import numpy as np

# --- CONFIGURATION ---
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, 'train') 
IMG_SIZE = 96         # MobileNetV2 standard minimum is 96x96
BATCH_SIZE = 32
EPOCHS = 10           # Transfer learning converges faster
MODEL_PATH = "models/plant_diseases_mobilenet.h5"
VALIDATION_SPLIT = 0.2 

# --- 1. DATA LOADING (AUTO SPLIT) ---
print(f"[INFO] Loading Training Data ({(1-VALIDATION_SPLIT)*100}%) from {TRAIN_DIR}...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123, 
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

print(f"[INFO] Loading Validation Data ({VALIDATION_SPLIT*100}%) from {TRAIN_DIR}...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123, 
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"[INFO] Detected {NUM_CLASSES} classes: {class_names}")

# Optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data Augmentation 
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# --- 2. MOBILENET V2 ARCHITECTURE ---
# We use alpha=0.35 for the smallest possible model size
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False, 
    weights='imagenet',
    alpha=0.35
)

# Freeze the base model to keep pre-trained weights
base_model.trainable = False

# Build the final model
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)

# MobileNetV2 expects inputs in range [-1, 1]
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)

# Pass through MobileNet
x = base_model(x, training=False)

# Classification Head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs, name="MobileNetV2_Tiny")
model.summary()

# --- 3. TRAIN ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --- 4. FINE TUNING (OPTIONAL BUT RECOMMENDED) ---
# Unfreeze the last few layers for better accuracy
print("[INFO] Fine-tuning model...")
base_model.trainable = True
# Fine-tune from this layer onwards
fine_tune_at = 100 
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5  # Additional epochs
)

# --- 5. SAVE ---
if not os.path.exists("models"): os.makedirs("models")
model.save(MODEL_PATH)
print(f"[SUCCESS] Model Saved to {MODEL_PATH}")

# Save class names for C++
with open("classes.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")
print("[INFO] Class names saved to classes.txt")
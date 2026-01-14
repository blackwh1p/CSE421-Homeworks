import os
import tensorflow as tf
from tensorflow.keras.utils import get_file, to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np

# Import the model architecture
from resnet import ResNet

# ---------------------------------------------------------
# Configuration and Paths
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_DIR is kept for reference
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")
MODEL_FILE = os.path.join(CURRENT_DIR, "model/resnet_mnist.h5")

# Create dataset directory if it doesn't exist
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Model Parameters
NUM_CLASSES = 10
DATA_SHAPE = (32, 32, 3) # ST Model expects 32x32 RGB
BATCH_SIZE = 32
EPOCHS = 5 

# ---------------------------------------------------------
# Data Preparation Functions
# ---------------------------------------------------------
def prepare_tensor(images, out_shape):
    """
    Resizes images to 32x32, converts to RGB, and normalizes pixel values.
    """
    # Add channel dimension (28, 28) -> (28, 28, 1)
    images = tf.expand_dims(images, axis=-1)
    
    # Resize to (32, 32)
    images = tf.image.resize(images, out_shape[:2])
    
    # Convert Grayscale to RGB (1 channel -> 3 channels)
    images = tf.repeat(images, 3, axis=-1)
    
    # Normalize to [0.0, 1.0]
    images = images / 255.0
    
    return images

# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Loading and preparing MNIST dataset...")
    
    # Load Data using default cache
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess Data
    train_images = prepare_tensor(train_images, DATA_SHAPE)
    test_images = prepare_tensor(test_images, DATA_SHAPE)

    # One-hot encoding for labels
    train_labels = to_categorical(train_labels, NUM_CLASSES)
    test_labels = to_categorical(test_labels, NUM_CLASSES)

    # ---------------------------------------------------------
    # Model Setup
    # ---------------------------------------------------------
    print("[INFO] Initializing ResNet model...")
    # Using parameters: depth=8, dropout=0.15
    model = ResNet(input_shape=DATA_SHAPE, classes=NUM_CLASSES, depth=8, dropout=0.15)

    # ---------------------------------------------------------
    # Transfer Learning Logic (New URL)
    # ---------------------------------------------------------
    # Updated URL provided by you (converted to raw/main for downloading)
    st_weights_url = "https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/image_classification/resnetv1/ST_pretrainedmodel_public_dataset/cifar10/resnet_v1_8_32_tfs/resnet_v1_8_32_tfs.h5"
    
    print("[INFO] Attempting to download/load pretrained weights...")

    try:
        # Try to download weights
        # We save it as 'resnet_v1_8_32_new.h5' to avoid conflict with old cached files
        pretrained_model_path = get_file(fname="resnet_v1_8_32_new.h5", origin=st_weights_url, cache_subdir="models")
        
        # Load weights
        model.load_weights(pretrained_model_path, by_name=True, skip_mismatch=True)
        print("[INFO] Pretrained weights loaded successfully.")
        
        # Freeze initial layers for Transfer Learning
        num_layers_to_train = len(model.layers) // 3
        for layer in model.layers[:num_layers_to_train]:
            layer.trainable = False
            
    except Exception as e:
        print(f"[WARNING] Could not download or load pretrained weights.")
        print(f"[INFO] Detailed error: {e}")
        print("[INFO] Switching to Standard Training (Training from Scratch).")
        
        for layer in model.layers:
            layer.trainable = True

    # Compile Model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    print("[INFO] Starting training...")
    model.fit(
        train_images, 
        train_labels, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(test_images, test_labels)
    )

    # Save Model
    model.save(MODEL_FILE)
    print(f"[INFO] Model saved to: {MODEL_FILE}")
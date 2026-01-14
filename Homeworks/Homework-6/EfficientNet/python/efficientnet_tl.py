import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import get_file, to_categorical
from tensorflow.keras.datasets import mnist

# Import architecture from efficientnet.py
from efficientnet import EfficientNet

# Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
MODEL_FILE = os.path.join(MODEL_DIR, "efficientnet_mnist.h5")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Hyperparameters
NUM_CLASSES = 10
DATA_SHAPE = (32, 32, 3)
BATCH_SIZE = 64
EPOCHS = 10

def prepare_data():
    print("[INFO] Loading and preprocessing MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    def preprocess(images):
        # Resize and convert to RGB
        images = tf.expand_dims(images, axis=-1)
        images = tf.image.resize(images, [32, 32])
        images = tf.image.grayscale_to_rgb(images)
        images = images / 255.0
        return images.numpy()

    return preprocess(x_train), to_categorical(y_train, 10), preprocess(x_test), to_categorical(y_test, 10)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = prepare_data()

    print("[INFO] Initializing EfficientNet...")
    # The build error happened here; the fix is now inside efficientnet.py
    model = EfficientNet(input_shape=DATA_SHAPE, classes=NUM_CLASSES)

    # STM32 AI Model Zoo Pretrained Weights
    st_url = "https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/image_classification/efficientnet/ST_pretrainedmodel_public_dataset/flowers/st_efficientnet_lc_v1_128_tfs/st_efficientnet_lc_v1_128_tfs.h5"
    
    print("[INFO] Downloading STM pretrained weights...")
    try:
        weights_path = get_file("st_effnet_pretrained.h5", st_url, cache_subdir="models")
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("[INFO] Successfully loaded pretrained weights.")
        
        # Freeze initial 70% of layers
        freeze_until = int(len(model.layers) * 0.7)
        for layer in model.layers[:freeze_until]:
            layer.trainable = False
    except Exception as e:
        print(f"[WARNING] Weight transfer failed: {e}. Training from scratch.")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("[INFO] Starting training...")
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

    model.save(MODEL_FILE)
    print(f"[SUCCESS] Model saved to {MODEL_FILE}")
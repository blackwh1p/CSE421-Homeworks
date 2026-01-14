import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tflite2cc import convert_tflite2cc

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model/efficientnet_mnist.h5")
OUTPUT_CC_PATH = os.path.join(CURRENT_DIR, "../hdr_cnn") 

# Load MNIST for calibration data (100 images are sufficient)
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

def representative_dataset():
    for i in range(100):
        # Must match training preprocessing exactly
        img = x_test[i]
        img = np.expand_dims(img, axis=-1)
        img = tf.image.resize(img, [32, 32])
        img = tf.image.grayscale_to_rgb(img)
        img = img / 255.0
        yield [np.expand_dims(img, axis=0).astype(np.float32)]

if __name__ == "__main__":
    print("[INFO] Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable Full Integer Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Force Int8 Input and Output
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("[INFO] Converting to TFLite (Int8)...")
    tflite_model = converter.convert()

    # Generate C++ files
    convert_tflite2cc(tflite_model, OUTPUT_CC_PATH)
    print(f"[SUCCESS] Int8 Model Size: {len(tflite_model)/1024:.2f} KB")
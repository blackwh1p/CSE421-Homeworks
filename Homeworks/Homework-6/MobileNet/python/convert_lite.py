import os 
import tensorflow as tf
from tensorflow import keras
from tflite2cc import convert_tflite2cc

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model/hdr_cnn.h5")
TFLITE_MODEL_PATH = os.path.join(CURRENT_DIR, "model/hdr_cnn.tflite")
TFLITE_EXPORT_PATH = os.path.join(CURRENT_DIR, "../hdr_cnn")

# 1. Load
print(f"[INFO] Loading model from {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)

# 2. Convert (Float32)
print("[INFO] Converting to TFLite (Float32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Save & Report
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)
    
size_kb = len(tflite_model) / 1024
print(f"[INFO] Model Size: {size_kb:.2f} KB")

# 4. C++ Generation
convert_tflite2cc(tflite_model, TFLITE_EXPORT_PATH)
print("[SUCCESS] C++ file generated.")
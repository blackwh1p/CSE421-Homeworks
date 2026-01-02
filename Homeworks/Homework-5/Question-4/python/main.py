import os
import tensorflow as tf
from tflite2cc import convert_tflite2cc
# Assuming your paths
MODEL_DIR = r"C:\Users\ASUS\Desktop\Fall-2025\CSE 421\hw5\Question-4\models"
model_path = os.path.join(MODEL_DIR, "temperature_pred_mlp.h5")

# 1. Load and Convert
model = tf.keras.models.load_model(model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optimization is highly recommended for regression on MCU
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 2. Save .tflite
tflite_path = os.path.join(MODEL_DIR, "temperature_pred_mlp.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

# 3. Export to C++ (temperature_pred_mlp.h and .cc)
export_path = os.path.join(MODEL_DIR, "temperature_pred_mlp")
convert_tflite2cc(tflite_model, export_path)

print("Conversion complete. Move the .h and .cc files to Mbed Studio.")
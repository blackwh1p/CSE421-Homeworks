import os
import tensorflow as tf
from keras.models import load_model

# CHANGE 1: Import locally instead of from "Common"
from tflite2cc import convert_tflite2cc

# CHANGE 2: Define paths locally to avoid "Models" module error
# Assuming your .h5 file is in the same folder as this script
KERAS_MODEL_DIR = "." 
TFLITE_EXPORT_DIR = "."

# Name of your model file
model_filename = "models/har_mlp.h5"
output_model_name = "../har_mlp"

# Check if model exists
model_path = os.path.join(KERAS_MODEL_DIR, model_filename)
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found. Please place {model_filename} in this directory.")
    exit()

print("Loading Keras model...")
model = load_model(model_path)

print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optional: Add optimizations here if needed
# converter.optimizations = [tf.lite.Optimize.DEFAULT] 
tflite_model = converter.convert()

print("Generating C++ files...")
convert_tflite2cc(tflite_model, os.path.join(TFLITE_EXPORT_DIR, output_model_name))

print("Conversion Complete.")
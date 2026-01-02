import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tflite2cc import convert_tflite2cc

# 1. SETUP PATHS - Adjust to match your folder structure exactly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: 'Models' must be capitalized if your folder is named 'Models'
MODEL_DIR = os.path.join(BASE_DIR, "models") 
EXPORT_DIR = os.path.join(BASE_DIR, "exported_models")

if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

model_path = os.path.join(MODEL_DIR, "hdr_mlp.keras")

# 2. LOAD AND CONVERT
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
else:
    # Load the trained Keras model
    model = load_model(model_path)
    
    # Create the TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization: Important for microcontrollers (Quantization)
    # This reduces weight size from 32-bit float to 8-bit int
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Perform conversion to TFLite flatbuffer format
    tflite_model = converter.convert()
    
    # Save the binary .tflite file
    tflite_file_path = os.path.join(MODEL_DIR, "hdr_mlp.tflite")
    with open(tflite_file_path, "wb") as f:
        f.write(tflite_model)
        
    # 3. EXPORT TO C++ ARRAY
    # This generates the .h and .cc files for STM32CubeIDE
    export_file_base = os.path.join(EXPORT_DIR, "hdr_mlp_model")
    convert_tflite2cc(tflite_model, export_file_base)
    
    print(f"Success! Model exported to {EXPORT_DIR}")
import os 
import tensorflow as tf
import numpy as np
from tflite2cc import convert_tflite2cc

# Local Path Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_PATH = os.path.join(CURRENT_DIR, "model/squeezenet_tl_mnist.h5")
TFLITE_EXPORT_PATH = os.path.join(CURRENT_DIR, "../hdr_cnn") # Will generate .h and .cpp

def representative_dataset():
    (_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    # Use 100 samples for calibration
    x_test = x_test[:100]
    x_test = tf.expand_dims(x_test, axis=-1)
    x_test = tf.repeat(x_test, 3, axis=-1)
    x_test = tf.image.resize(x_test, [32, 32])
    x_test = x_test / 255.0
    for i in range(100):
        yield [x_test[i:i+1]]

if __name__ == "__main__":
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"[ERROR] Run squeezenet_tl.py first. Missing: {KERAS_MODEL_PATH}")
        exit(1)

    hdr_cnn_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(hdr_cnn_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset
    
    # Set IO to Uint8 for easier UART handling
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    print("[INFO] Converting to Full Int8 TFLite...")
    tflite_model = converter.convert()

    # Save .tflite file
    with open(KERAS_MODEL_PATH.replace(".h5", ".tflite"), "wb") as f:
        f.write(tflite_model)

    # Export to C++
    convert_tflite2cc(tflite_model, TFLITE_EXPORT_PATH)
    print(f"[SUCCESS] C++ files generated at {TFLITE_EXPORT_PATH}.cpp/h")
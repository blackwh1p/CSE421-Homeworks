import os
import tensorflow as tf

# ---------------------------------------------------------
# Configuration and Paths
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model/resnet_mnist.h5")
OUTPUT_CC_PATH = os.path.join(CURRENT_DIR, "../hdr_cnn.cpp")

# ---------------------------------------------------------
# C Array Generation Function
# ---------------------------------------------------------
def convert_to_c_array(tflite_blob, filename):
    """
    Writes the TFLite binary blob to a C++ source file as a byte array.
    """
    hex_array = []
    for i, val in enumerate(tflite_blob):
        hex_array.append(f"0x{val:02x}")
    
    c_code = (
        '#include "hdr_cnn.h"\n\n'
        f'// Model Size: {len(tflite_blob)} bytes\n'
        f'const unsigned int converted_model_tflite_len = {len(tflite_blob)};\n'
        'const unsigned char converted_model_tflite[] __attribute__((aligned(8))) = {\n'
    )
    
    # Write in chunks of 12 bytes per line
    for i in range(0, len(hex_array), 12):
        line = ", ".join(hex_array[i:i+12])
        c_code += "  " + line + ",\n"
    
    c_code += "};\n"

    with open(filename, 'w') as f:
        f.write(c_code)
    print(f"[SUCCESS] C file generated: {filename}")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at {MODEL_PATH}. Please run resnet_tl.py first.")
        exit(1)

    print("[INFO] Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Initialize TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    print("[INFO] Converting to TFLite with quantization...")
    tflite_model = converter.convert()

    # Convert to C Array for Mbed
    print("[INFO] Generating C array...")
    convert_to_c_array(tflite_model, OUTPUT_CC_PATH)
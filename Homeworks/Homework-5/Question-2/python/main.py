import os
import tensorflow as tf
from tensorflow import keras

# --- 1. The Conversion Logic (Taken from your tflite2cc.py) ---
def convert_tflite2cc(tflite_model, c_out_path):
    """
    Converts TFLite binary to C++ source/header pair.
    Output Variable Name: converted_model_tflite
    """
    # Read model if passed as path
    if isinstance(tflite_model, str):
        with open(tflite_model, "rb") as tflite_file:
            tflite_model = tflite_file.read()

    # Define output file names
    hdr_filepath = c_out_path + ".h"
    src_filepath = c_out_path + ".cpp"
    hdr_filename = os.path.basename(hdr_filepath)
    
    arr_len = len(tflite_model)

    # Write Header (.h)
    with open(hdr_filepath, "w") as hdr_file:
        hdr_file.write("/* Auto-generated header */\n")
        hdr_file.write(f"#ifndef {os.path.basename(c_out_path).upper()}_H\n")
        hdr_file.write(f"#define {os.path.basename(c_out_path).upper()}_H\n\n")
        # IMPORTANT: This matches the variable name in your tflite2cc.py
        hdr_file.write("extern const unsigned char converted_model_tflite[];\n")
        hdr_file.write("extern const unsigned int converted_model_tflite_len;\n\n")
        hdr_file.write("#endif\n")

    # Write Source (.cpp)
    with open(src_filepath, "w") as src_file:
        src_file.write(f'#include "{hdr_filename}"\n\n')
        src_file.write("const unsigned char converted_model_tflite[] = {\n")
        
        # Write hex data
        hex_lines = [", ".join([f"0x{b:02x}" for b in tflite_model[i:i+15]]) 
                     for i in range(0, arr_len, 15)]
        src_file.write(",\n".join(hex_lines))
        
        src_file.write(f"\n}};\n\nconst unsigned int converted_model_tflite_len = {arr_len};\n")

    print(f"SUCCESS! Generated:\n  - {hdr_filepath}\n  - {src_filepath}")


# --- 2. Main Execution (Replaces your main.py) ---
if __name__ == "__main__":
    # Get current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    model_path = os.path.join(current_dir, "models/kws_mlp.h5")
    output_base_name = os.path.join(current_dir, "../kws_mlp") # Output files: kws_mlp.h, kws_mlp.cpp

    if not os.path.exists(model_path):
        print(f"ERROR: Could not find '{model_path}'")
        print("Please ensure 'kws_mlp.h5' is in this folder.")
    else:
        print(f"Loading model: {model_path}")
        model = keras.models.load_model(model_path)

        print("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        print("Generating C++ files...")
        convert_tflite2cc(tflite_model, output_base_name)
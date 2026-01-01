import os

def convert_tflite2cc(tflite_model, output_base_path):
    """
    Converts a tflite model byte array to C++ source and header files.
    """
    
    # Extract filename from path
    model_name = os.path.basename(output_base_path)
    
    header_file = output_base_path + ".h"
    source_file = output_base_path + ".cc"
    
    # 1. Generate Header File (.h)
    header_content = f"""
#ifndef {model_name.upper()}_H_
#define {model_name.upper()}_H_

extern const unsigned char {model_name}[];
extern const int {model_name}_len;

#endif
"""
    with open(header_file, 'w') as f:
        f.write(header_content)

    # 2. Generate Source File (.cc)
    # Convert bytes to hex string
    hex_array = []
    for i, val in enumerate(tflite_model):
        hex_array.append(f"0x{val:02x}")

    # Format into lines
    hex_str = ""
    for i in range(0, len(hex_array), 12):
        hex_str += "  " + ", ".join(hex_array[i:i+12]) + ",\n"

    source_content = f"""
#include "{model_name}.h"

const unsigned char {model_name}[] = {{
{hex_str}
}};
const int {model_name}_len = {len(tflite_model)};
"""
    with open(source_file, 'w') as f:
        f.write(source_content)
    
    print(f"Generated: {header_file}")
    print(f"Generated: {source_file}")
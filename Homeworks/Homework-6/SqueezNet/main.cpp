#include "mbed.h"
#include "lib_model.h"
#include "hdr_cnn.h"

// Buffers must be int8_t for quantized models
const int IMG_SIZE = 32 * 32 * 3;
static uint8_t rx_buffer[IMG_SIZE];
static int8_t input_int8[IMG_SIZE]; 
static int8_t output_int8[10];      

static BufferedSerial pc(USBTX, USBRX);

int main() {
    pc.set_baud(115200);
    
    // Initialize
    if (InitModel(converted_model_tflite, nullptr, 0) != MODEL_OK) {
        return -1;
    }

    while (true) {
        // Read data from UART
        ssize_t read_bytes = 0;
        while (read_bytes < IMG_SIZE) {
            if (pc.readable()) read_bytes += pc.read(rx_buffer + read_bytes, IMG_SIZE - read_bytes);
        }

        // Map [0, 255] to [-128, 127] for Int8 model
        for(int i=0; i<IMG_SIZE; i++) {
            input_int8[i] = (int8_t)((int)rx_buffer[i] - 128);
        }

        // CALL THE CORRECT FUNCTION NAME HERE
        if (RunInferenceInt8(input_int8, output_int8) == MODEL_OK) {
            // Find max class
            int8_t max_idx = 0;
            int8_t max_val = -128;
            for(int8_t i=0; i<10; i++) {
                if(output_int8[i] > max_val) { 
                    max_val = output_int8[i]; 
                    max_idx = i; 
                }
            }
            // Send back 1 byte prediction
            pc.write(&max_idx, 1);
        }
    }
}
#include "mbed.h"
#include "lib_model.h"
#include "hdr_cnn.h"

// --------------------------------------------------------------------------
// CONFIGURATION (Matches resnet_tl.py DATA_SHAPE)
// --------------------------------------------------------------------------
const int IMG_WIDTH = 32;
const int IMG_HEIGHT = 32;
const int IMG_CHANNELS = 3; // RGB
const int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS; // 3072 bytes

// Buffers
uint8_t rx_buffer[IMG_SIZE];
float output_data[10];

// Serial Object
static BufferedSerial pc(USBTX, USBRX);

int main()
{
    // Baud rate match
    pc.set_baud(115200); 

    // 1. Initialize Model
    // We pass nullptr because memory is managed inside lib_model.cpp
    int init_status = InitModel(converted_model_tflite, nullptr, 0);
    
    if (init_status != MODEL_OK) {
        // Error Indicator: Fast Blink
        DigitalOut led(LED1);
        while(1) { led = !led; thread_sleep_for(100); }
    }

    // 2. Main Loop
    while (true)
    {
        // A. Read raw image data (Blocking)
        ssize_t total_bytes_read = 0;
        while (total_bytes_read < IMG_SIZE) {
            if (pc.readable()) {
                total_bytes_read += pc.read(rx_buffer + total_bytes_read, IMG_SIZE - total_bytes_read);
            }
        }

        // B. Preprocess (Uint8 -> Float)
        // Normalize [0-255] to [0.0-1.0] matching resnet_tl.py logic
        static float input_float_buffer[IMG_SIZE];
        for(int i=0; i<IMG_SIZE; i++) {
            input_float_buffer[i] = (float)rx_buffer[i] / 255.0f;
        }

        // C. Run Inference
        RunInference(input_float_buffer, output_data);

        // D. Find Max Probability
        float max_score = -1.0f;
        int8_t predicted_class = -1;

        for (int i = 0; i < 10; i++) {
            if (output_data[i] > max_score) {
                max_score = output_data[i];
                predicted_class = (int8_t)i;
            }
        }

        // E. Send Result (1 Byte ONLY)
        pc.write(&predicted_class, 1);
    }
}
#include "mbed.h"
#include "lib_model.h"
#include "hdr_cnn.h"

// --------------------------------------------------------------------------
// CONFIGURATION
// --------------------------------------------------------------------------
// Image Dimensions: 32x32 RGB => 32 * 32 * 3 = 3072 bytes
const int IMG_WIDTH = 32;
const int IMG_HEIGHT = 32;
const int IMG_CHANNELS = 3;
const int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;

// Buffers
// rx_buffer: Received raw bytes (0-255) from PC via UART
uint8_t rx_buffer[IMG_SIZE];
// output_data: Probabilities for 10 digits
float output_data[10];

// Serial Object (USBTX/RX connects to ST-Link Virtual COM Port)
static BufferedSerial pc(USBTX, USBRX);

int main()
{
    // Set baud rate to match Python script (115200)
    pc.set_baud(115200); 

    // 1. Initialize Model
    // We pass nullptr because memory is managed inside lib_model.cpp
    int init_status = InitModel(converted_model_tflite, nullptr, 0);
    
    if (init_status != MODEL_OK) {
        // Blink LED fast to indicate initialization error
        DigitalOut led(LED1);
        while(1) { led = !led; thread_sleep_for(100); }
    }

    // 2. Main Loop: Wait for Data -> Predict -> Send Result
    while (true)
    {
        // A. Read raw image data from UART (Blocking call)
        // We wait until exactly 3072 bytes are received
        ssize_t total_bytes_read = 0;
        while (total_bytes_read < IMG_SIZE) {
            if (pc.readable()) {
                total_bytes_read += pc.read(rx_buffer + total_bytes_read, IMG_SIZE - total_bytes_read);
            }
        }

        // B. Preprocess Data (Uint8 [0-255] -> Float [0.0-1.0])
        // We convert the raw bytes to normalized floats for the model
        static float input_float_buffer[IMG_SIZE];
        for(int i=0; i<IMG_SIZE; i++) {
            input_float_buffer[i] = (float)rx_buffer[i] / 255.0f;
        }

        // C. Run Inference
        RunInference(input_float_buffer, output_data);

        // D. Find Max Probability Class
        float max_score = -1.0f;
        int8_t predicted_class = -1; // int8_t to send as a single byte

        for (int i = 0; i < 10; i++) {
            if (output_data[i] > max_score) {
                max_score = output_data[i];
                predicted_class = (int8_t)i;
            }
        }

        // E. Send Result back to PC (1 Byte)
        pc.write(&predicted_class, 1);
    }
}
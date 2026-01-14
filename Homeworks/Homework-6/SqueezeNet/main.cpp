#include "mbed.h"
#include "lib_model.h"
#include "hdr_cnn.h"

// Buffers must be static to avoid stack growth
static uint8_t rx_buffer[3072]; 
static int8_t output_buffer[10];
static BufferedSerial pc(USBTX, USBRX);
DigitalOut led(LED1);

int main() {
    pc.set_baud(115200);
    
    // Initialize model without any serial logging
    if (InitModel(converted_model_tflite, nullptr, 0) != MODEL_OK) {
        // Blink fast if Init fails
        while(1) { led = !led; thread_sleep_for(100); }
    }

    while (true) {
        ssize_t total_read = 0;
        // Wait for 3072 bytes (32x32x3)
        while (total_read < 3072) {
            if (pc.readable()) {
                total_read += pc.read(rx_buffer + total_read, 3072 - total_read);
            }
        }

        // Run inference with the signed pointer cast
        if (RunInferenceInt8((int8_t*)rx_buffer, output_buffer) == MODEL_OK) {
            int8_t max_idx = 0;
            int8_t max_val = -128; 

            for(int i = 0; i < 10; i++) {
                if(output_buffer[i] > max_val) { 
                    max_val = output_buffer[i]; 
                    max_idx = (int8_t)i; 
                }
            }
            
            // Send 1-byte result via UART
            pc.write(&max_idx, 1);
            led = !led; 
        }
    }
}
/*
 * FOMO Object Detection - Multi-Object Reporting Version
 * Target Board: STM32F746G-Discovery
 * Framework: TensorFlow Lite for Microcontrollers
 * 
 * Protocol:
 * 1. Board sends '#' (Ready)
 * 2. PC sends image data (96x96 bytes)
 * 3. Board runs inference
 * 4. Board sends "DET:x,y,score\n" for every detected object
 * 5. Board sends "END:count\n" to finish
 */

#include "mbed.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h" 

// --- HARDWARE SETTINGS ---
// USBTX and USBRX are the pins for the ST-Link Virtual COM Port
static UnbufferedSerial pc(USBTX, USBRX, 115200);
DigitalOut led(LED1);

// --- MEMORY SETTINGS ---
// Tensor Arena for TensorFlow allocations
// 250KB is sufficient for MobileNetV2 alpha 0.35
const int kTensorArenaSize = 250 * 1024; 
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// --- CONSTANTS ---
// FOMO output is a 12x12 grid for 96x96 input
const int GRID_WIDTH = 12;
const int GRID_HEIGHT = 12;

// Threshold for detection (int8 range: -128 to 127)
// -50 is approx 0.3 probability. Adjust higher (e.g., 0) to reduce false positives.
const int8_t SCORE_THRESHOLD = 20; 

// Helper function to send string over serial
void serial_write_str(const char* str) {
    pc.write(str, strlen(str));
}

int main() {
    // Startup Blink
    led = 1; thread_sleep_for(200); led = 0; thread_sleep_for(200);
    
    // --- TFLITE SETUP ---
    // Resolver: Registers the operators used by the model
    static tflite::MicroMutableOpResolver<20> resolver;
    
    // Convolution & Geometry Ops
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddReshape();
    resolver.AddPad();
    resolver.AddConcatenation(); // Important for MobileNet blocks
    
    // Activation Ops
    resolver.AddRelu();
    resolver.AddRelu6();         // Critical: MobileNetV2 uses ReLU6
    resolver.AddSoftmax();
    resolver.AddLogistic();      // Sigmoid (Last layer for FOMO)
    
    // Math Ops
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddMean();
    resolver.AddFullyConnected();
    
    // Quantization Ops
    resolver.AddQuantize();
    resolver.AddDequantize();
    
    // Load Model
    const tflite::Model* model = tflite::GetModel(fruitfly_fomo_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        serial_write_str("Error: Model Schema Version Mismatch!\n");
        while(1) { led = !led; thread_sleep_for(50); }
    }

    // Instantiate Interpreter
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    // Allocate Tensors
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        serial_write_str("Error: AllocateTensors() Failed! Increase Arena Size.\n");
        while(1) { led = !led; thread_sleep_for(500); }
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    serial_write_str("--- STM32 FOMO Multi-Detect Ready ---\n");

    // --- MAIN LOOP ---
    while (true) {
        // 1. Send Handshake
        char handshake = '#';
        pc.write(&handshake, 1);

        // 2. Receive Image
        // Input expects signed int8 [-128, 127]
        int total_bytes = input->bytes;
        int bytes_received = 0;
        int8_t* input_ptr = input->data.int8; 
        
        while (bytes_received < total_bytes) {
            if (pc.readable()) {
                pc.read(&input_ptr[bytes_received], 1);
                bytes_received++;
            }
        }

        led = 1; // Processing indicator ON
        
        // 3. Run Inference
        TfLiteStatus invoke_status = interpreter.Invoke();
        if (invoke_status != kTfLiteOk) {
            serial_write_str("Error: Invoke Failed!\n");
            led = 0;
            continue;
        }

        // 4. Process Results & Send Multiple Detections
        int8_t* results = output->data.int8;
        int detection_count = 0;
        char buffer[64];
        
        // Iterate through the 12x12 grid
        for (int y = 0; y < GRID_HEIGHT; y++) {
            for (int x = 0; x < GRID_WIDTH; x++) {
                // Calculate index in 1D array
                // If model has multiple classes, index = (y * width + x) * num_classes + class_id
                int index = y * GRID_WIDTH + x;
                int8_t score = results[index];
                
                // If confidence is above threshold, report it
                if (score > SCORE_THRESHOLD) {
                    // Format: DET:x,y,score
                    int len = sprintf(buffer, "DET:%d,%d,%d\n", x, y, score);
                    pc.write(buffer, len);
                    detection_count++;
                }
            }
        }

        // 5. Send End Signal
        int end_len = sprintf(buffer, "END:%d\n", detection_count);
        pc.write(buffer, end_len);
        
        led = 0; // Processing indicator OFF
    }
}
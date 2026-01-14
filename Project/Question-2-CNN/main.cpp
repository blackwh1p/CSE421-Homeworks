/*
 * Final Project - Question 2
 * Model: MobileNetV2 (0.35 alpha, 96x96 input)
 * Task: Tomato Disease/Pest Classification (16 Classes)
 * Platform: STM32F746G-Discovery via Mbed Studio
 */

#include "mbed.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h" 

// --- HARDWARE ---
static UnbufferedSerial pc(USBTX, USBRX, 115200);
DigitalOut led(LED1);

// --- MEMORY ---
const int kTensorArenaSize = 260 * 1024; 
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// --- MODEL SETTINGS ---
const int IMAGE_WIDTH = 96;
const int IMAGE_HEIGHT = 96;
const int IMAGE_CHANNELS = 3;

// Updated for the new dataset (16 Classes)
const int NUM_CLASSES = 16;

void serial_write_str(const char* str) {
    pc.write(str, strlen(str));
}

int main() {
    led = 1; thread_sleep_for(200); led = 0; thread_sleep_for(200);
    
    // 1. Load Model
    const tflite::Model* model = tflite::GetModel(mobilenet_model);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        serial_write_str("[ERROR] Model Schema Version Mismatch!\n");
        while(1) { led = !led; thread_sleep_for(50); }
    }

    // 2. Register Ops
    static tflite::MicroMutableOpResolver<20> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddMean();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddReshape();
    resolver.AddPad();
    resolver.AddConcatenation();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddLogistic();

    // 3. Instantiate Interpreter
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    // 4. Allocate Memory
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        serial_write_str("[ERROR] AllocateTensors Failed!\n");
        while(1) { led = !led; thread_sleep_for(100); }
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    serial_write_str("--- Tomato Disease Classifier Ready ---\n");

    while (true) {
        // Handshake
        char handshake = '#';
        pc.write(&handshake, 1);

        // Receive Image
        int total_bytes = input->bytes;
        int bytes_received = 0;
        int8_t* input_ptr = input->data.int8; 
        
        while (bytes_received < total_bytes) {
            if (pc.readable()) {
                pc.read(&input_ptr[bytes_received], 1);
                bytes_received++;
            }
        }

        led = 1; 
        interpreter.Invoke();
        led = 0;

        // Process Output
        int8_t* results = output->data.int8;
        int max_index = 0;
        int8_t max_score = -128; 

        for (int i = 0; i < NUM_CLASSES; i++) {
            if (results[i] > max_score) {
                max_score = results[i];
                max_index = i;
            }
        }

        // Send Result: "CLASS:index,score"
        char buffer[64];
        int len = sprintf(buffer, "CLASS:%d,%d\n", max_index, max_score);
        pc.write(buffer, len);
    }
}
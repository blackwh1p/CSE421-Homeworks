/* * Application 3: Handwritten Digit Recognition (12.9)
 * Mode: Data Injection (Pre-extracted features from PC)
 * Files required: lib_model.h/cpp, lib_serial.h/cpp, hdr_mlp.h
 */

#include "mbed.h"
#include "lib_model.h"
#include "lib_serial.h"

// The model file you generated with your Python script
#include "hdr_mlp.h" 

// --- Constants ---
#define NUM_FEATURES 7    // The 7 Hu Moments (Input layer)
#define NUM_CLASSES 10     // Digits 0-9 (Output layer)

// --- Globals ---
// Buffer to hold the 7 floats received from the Python script
float feature_buffer[NUM_FEATURES];

// TensorFlow Lite Micro Tensors
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor Arena: 60KB is plenty for the MLP model used in the book
constexpr int kTensorArenaSize = 60 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

int main() {
    // 1. Initialize Serial Communication 
    // This matches the lib_serial.cpp implementation in the repo
    LIB_SERIAL_Init(); 

    // 2. Initialize the Neural Network Model
    // Important: 'hdr_mlp' must match the array name inside your hdr_mlp.h
    // If your generated file uses a different name, update it here.
    if (LIB_MODEL_Init(hdr_mlp, &input, tensor_arena, kTensorArenaSize) != 0) {
        // Initialization error
        return -1;
    }

    while (true) {
        // ---------------------------------------------------------
        // A. Receive Data
        // ---------------------------------------------------------
        // MCU waits here for the Python script to send 7 float32 values
        LIB_SERIAL_Receive(feature_buffer, NUM_FEATURES, TYPE_F32);

        // ---------------------------------------------------------
        // B. Prepare Input
        // ---------------------------------------------------------
        // Feed the received Hu Moments into the model's input tensor
        for (int i = 0; i < NUM_FEATURES; i++) {
            input->data.f[i] = feature_buffer[i];
        }

        // ---------------------------------------------------------
        // C. Run Inference
        // ---------------------------------------------------------
        // Run the interpreter to get predictions
        LIB_MODEL_Run(&output);

        // ---------------------------------------------------------
        // D. Transmit Results
        // ---------------------------------------------------------
        // Send the 10 probabilities back to the PC for display
        LIB_SERIAL_Transmit(output->data.f, NUM_CLASSES, TYPE_F32);
    }
}
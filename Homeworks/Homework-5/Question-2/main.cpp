/*
 * main.cpp
 * FIXED: Adds Peak Detection to align audio before inference
 */
#include "mbed.h"
#include "ks_feature_extraction.h"
#include "lib_serial.h"
#include "lib_model.h"
#include "kws_mlp.h" 
#include <cmath> 

// --- Peripherals ---
UnbufferedSerial pc(USBTX, USBRX); 

// --- CONFIGURATION ---
#define BUFFER_SIZE     (32000) 
// Window size for feature extraction (1024 samples)
#define WINDOW_SIZE     (1024)

int16_t AudioBuffer[BUFFER_SIZE]; 
int16_t AudioBufferDown[BUFFER_SIZE/4];
float32_t AudioBufferF32Down[BUFFER_SIZE/4];
float32_t ExtractedFeatures[nbDctOutputs * 2];

// Tensor Arena
constexpr int kTensorArenaSize = 60 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// --- Helper: Receive Audio from PC ---
void ReceiveAudioFromPC(int16_t* buffer, int buffer_len) {
    printf("WAITING_FOR_AUDIO\r\n"); 
    int bytes_received = 0;
    int total_bytes = buffer_len * sizeof(int16_t);
    uint8_t* byte_ptr = (uint8_t*)buffer;

    while (bytes_received < total_bytes) {
        if (pc.readable()) {
            pc.read(&byte_ptr[bytes_received], 1);
            bytes_received++;
        }
    }
}

int main()
{
    pc.baud(115200);
    printf("--- KWS Mbed Started (Peak Align Fix) ---\r\n");

    ks_mfcc_init();

    if (LIB_MODEL_Init(converted_model_tflite, &input, tensor_arena, kTensorArenaSize) != 0) {
        printf("Model Init Failed!\r\n");
        while(1);
    }

    while (true)
    {
        // 1. Receive Data
        ReceiveAudioFromPC(AudioBuffer, BUFFER_SIZE);

        int16_t max_val = 0; 
        int i = 0, j = 0;
        int down_size = BUFFER_SIZE/4;

        // 2. Downsample (32kHz -> 8kHz)
        for (i = 0; i < down_size; ++i)
        {
            AudioBufferDown[i] = AudioBuffer[j];
            j = j + 4; 
        }

        // 3. Normalize & Convert to Float
        // Also find the loudest point (Peak) to center the window
        max_val = 0;
        int peak_index = 0;

        for (int k = 0; k < down_size; k++) {
            int16_t val = abs(AudioBufferDown[k]);
            if (val > max_val) {
                max_val = val;
                peak_index = k;
            }
        }
        if (max_val == 0) max_val = 1; 
        
        for (i = 0; i < down_size; ++i)
        {
            AudioBufferF32Down[i] = (float32_t)AudioBufferDown[i]/(float32_t)max_val;
        }

        // 4. Determine Extraction Start Point
        // We need 2 windows of 1024 samples (Total 2048 samples).
        // Let's try to center the peak in this 2048 window.
        // Start = Peak - 1024.
        int start_idx = peak_index - 1024;

        // Boundary checks
        if (start_idx < 0) start_idx = 0;
        if (start_idx + 2048 > down_size) start_idx = down_size - 2048;
        
        // Safety check if buffer is too small
        if (start_idx < 0) start_idx = 0; 

        // 5. Feature Extraction (Aligned to Peak)
        // Window 1
        ks_mfcc_extract_features(&AudioBufferF32Down[start_idx], ExtractedFeatures);
        // Window 2
        ks_mfcc_extract_features(&AudioBufferF32Down[start_idx + 1024], &ExtractedFeatures[nbDctOutputs]);

        memcpy(input->data.f, ExtractedFeatures, nbDctOutputs * 2 * sizeof(float));
        LIB_MODEL_Run(&output);

        // 7. Send Results (Now works correctly with Python!)
        LIB_SERIAL_Transmit(output->data.f, 10, TYPE_F32);

        printf("Inference Done.\r\n");
    }
}
#include "mbed.h"
#include "arm_math.h"
#include "har_feature_extraction.h"
#include "bayes_cls_inference.h"
#include "bayes_cls_config.h"

#define BAUD_RATE 115200
#define WINDOW_SIZE 64   // Matches Python
#define NUM_AXES 3

UnbufferedSerial pc(USBTX, USBRX);

// Buffers
float32_t acc_data[3][WINDOW_SIZE]; 
HAR_FtrExtOutput feature_out;
float32_t feature_vector[NUM_FEATURES];
float32_t class_probs[NUM_CLASSES];

// Matrix Wrappers
arm_matrix_instance_f32 mat_input;
arm_matrix_instance_f32 mat_output;

int main() {
    pc.baud(BAUD_RATE);
    
    // --- FIX: Matrix Dimensions for Book Library ---
    // The library computes: Transpose(Input) * Covariance
    // Input (10x1) -> Transposed to (1x10)
    // (1x10) * (10x10) = (1x10) -> OK
    mat_input.numRows = NUM_FEATURES; // 10
    mat_input.numCols = 1;            // 1
    mat_input.pData = feature_vector;

    mat_output.numRows = 1;
    mat_output.numCols = NUM_CLASSES;
    mat_output.pData = class_probs;

    while (true) {
        // 1. Sync
        char ready = 'R';
        pc.write(&ready, 1);

        // 2. Receive Data
        char* ptr = (char*)acc_data;
        int bytes_remaining = sizeof(acc_data);
        while (bytes_remaining > 0) {
            if (pc.readable()) {
                int n = pc.read(ptr, bytes_remaining);
                bytes_remaining -= n;
                ptr += n;
            }
        }

        // 3. Feature Extraction
        har_extract_features(acc_data, &feature_out);

        // 4. Fill Feature Vector (ORDER MATTERS)
        // Python adds Mean columns first
        feature_vector[0] = feature_out.x_mean;
        feature_vector[1] = feature_out.y_mean;
        feature_vector[2] = feature_out.z_mean;
        
        // Then Positive Count columns
        feature_vector[3] = feature_out.x_pos;
        feature_vector[4] = feature_out.y_pos;
        feature_vector[5] = feature_out.z_pos;
        
        // Then FFT Std Dev columns
        feature_vector[6] = feature_out.fft_sd_x;
        feature_vector[7] = feature_out.fft_sd_y;
        feature_vector[8] = feature_out.fft_sd_z;
        
        // Finally SMA
        feature_vector[9] = feature_out.sma;

        // 5. Inference
        bayes_cls_predict(&mat_input, &mat_output);

        // 6. Argmax
        int best_class = 0;
        float max_val = class_probs[0];
        for(int i=1; i<NUM_CLASSES; i++) {
            if (class_probs[i] > max_val) {
                max_val = class_probs[i];
                best_class = i;
            }
        }

        char result = (char)best_class;
        pc.write(&result, 1);
    }
}
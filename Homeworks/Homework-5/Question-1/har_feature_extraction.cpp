#include "har_feature_extraction.h"

// Buffers for FFT calculations
static float fft_input[VECTOR_LEN];
static float fft_output[VECTOR_LEN]; // RFFT output size is same as input in CMSIS 
static float fft_mag[VECTOR_LEN/2 + 1];

// CMSIS-DSP RFFT Instance
static arm_rfft_fast_instance_f32 fft_instance;
static bool fft_initialized = false;

int8_t har_extract_features(float acc_data[3][VECTOR_LEN], HAR_FtrExtOutput *output) {
    if (!fft_initialized) {
        arm_rfft_fast_init_f32(&fft_instance, VECTOR_LEN);
        fft_initialized = true;
    }

    float sma_accum = 0.0f;
    
    // 1. Calculate Means and Pos Counts
    float means[3] = {0};
    float pos_counts[3] = {0};
    
    for (int axis = 0; axis < 3; axis++) {
        for (int i = 0; i < VECTOR_LEN; i++) {
            means[axis] += acc_data[axis][i];
            if (acc_data[axis][i] > 0) pos_counts[axis] += 1.0f;
        }
        means[axis] /= VECTOR_LEN;
    }
    
    output->x_mean = means[0];
    output->y_mean = means[1];
    output->z_mean = means[2];
    output->x_pos = pos_counts[0];
    output->y_pos = pos_counts[1];
    output->z_pos = pos_counts[2];

    // 2. FFT Based Features (Std Dev & SMA)
    float fft_stds[3] = {0};

    for (int axis = 0; axis < 3; axis++) {
        // Copy data to fft input buffer
        memcpy(fft_input, acc_data[axis], VECTOR_LEN * sizeof(float));
        
        // Perform RFFT
        arm_rfft_fast_f32(&fft_instance, fft_input, fft_output, 0);
        
        // Calculate Magnitude (Complex to Real)
        arm_cmplx_mag_f32(fft_output, fft_mag, VECTOR_LEN/2 + 1);
        
        // Calculate Standard Deviation of magnitudes (excluding DC at index 0)
        float std_val;
        float mean_val; // unused
        arm_std_f32(&fft_mag[1], VECTOR_LEN/2, &std_val); // Use block size VECTOR_LEN/2
        fft_stds[axis] = std_val;
        
        // Accumulate SMA (Sum of magnitudes)
        for(int k=1; k <= VECTOR_LEN/2; k++) {
            sma_accum += fft_mag[k];
        }
    }

    output->fft_sd_x = fft_stds[0];
    output->fft_sd_y = fft_stds[1];
    output->fft_sd_z = fft_stds[2];
    
    // Normalization factor for SMA (as per book example logic)
    output->sma = sma_accum / (VECTOR_LEN/2);

    return 0;
}
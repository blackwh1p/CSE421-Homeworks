#ifndef HAR_FEATURE_EXTRACTION_H
#define HAR_FEATURE_EXTRACTION_H

#include "mbed.h"
#include "arm_math.h"

#define VECTOR_LEN 80 

typedef struct {
    float x_mean; float y_mean; float z_mean;
    float x_pos;  float y_pos;  float z_pos;
    float fft_sd_x; float fft_sd_y; float fft_sd_z;
    float sma;
} HAR_FtrExtOutput;

int8_t har_extract_features(float acc_data[3][VECTOR_LEN], HAR_FtrExtOutput *output);

#endif
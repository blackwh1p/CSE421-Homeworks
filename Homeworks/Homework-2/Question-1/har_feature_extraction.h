/*
 * har_feature_extraction.h
 */

#ifndef INC_HAR_FEATURE_EXTRACTION_H_
#define INC_HAR_FEATURE_EXTRACTION_H_

// --- STEP 1: Add this block ---
#ifdef __cplusplus
extern "C" {
#endif
// ------------------------------

#include <stdint.h>

#ifdef STM32F746xx
#include "stm32f746xx.h"
#endif

#include "arm_math.h"

#define VECTOR_LEN	64

typedef struct __HAR_FtrExtOutput
{
	float x_mean;
	float y_mean;
	float z_mean;
	float x_pos;
	float y_pos;
	float z_pos;
	float fft_sd_x;
	float fft_sd_y;
	float fft_sd_z;
	float sma;
} HAR_FtrExtOutput;

// This is the function causing the error
int8_t har_extract_features(float32_t acc_data[3][VECTOR_LEN], HAR_FtrExtOutput *output);

// --- STEP 2: Add this block ---
#ifdef __cplusplus
}
#endif
// ------------------------------

#endif /* INC_HAR_FEATURE_EXTRACTION_H_ */
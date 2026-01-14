#ifndef LIB_MODEL_H
#define LIB_MODEL_H

#include <stdint.h>

// Status Codes
#define MODEL_OK 0
#define MODEL_ERROR -1

// Function Prototypes

/**
 * @brief Initializes the model. 
 * Note: optional_arena_ptr and optional_size are ignored if lib_model manages its own memory.
 */
int InitModel(const unsigned char* model_data, uint8_t* optional_arena_ptr, int optional_size);

/**
 * @brief Runs inference.
 */
int RunInference(float* input_data, float* output_data);

#endif // LIB_MODEL_H
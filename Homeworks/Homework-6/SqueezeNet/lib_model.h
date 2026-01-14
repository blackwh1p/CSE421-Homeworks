#ifndef LIB_MODEL_H
#define LIB_MODEL_H

#include <stdint.h>

#define MODEL_OK 0
#define MODEL_ERROR -1

int InitModel(const unsigned char* model_data, uint8_t* optional_arena_ptr, int optional_size);
int RunInferenceInt8(int8_t* input_data, int8_t* output_data);

#endif
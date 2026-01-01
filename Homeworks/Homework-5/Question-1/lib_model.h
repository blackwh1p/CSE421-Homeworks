#ifndef LIB_MODEL_H
#define LIB_MODEL_H

#include "mbed.h"
// CHANGE: Use mutable resolver instead of all ops
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

int8_t LIB_MODEL_Init(const unsigned char *model_data, TfLiteTensor **input, uint8_t *tensor_arena, int arena_size);
int8_t LIB_MODEL_Run(TfLiteTensor **output);

#endif
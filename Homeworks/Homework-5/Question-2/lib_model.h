/*
 * lib_model.h
 * Mbed compatible version
 */
#ifndef INC_LIB_MODEL_H_
#define INC_LIB_MODEL_H_

// Include TFLite Micro headers
// These path locations are standard for the 'tensorflow' Mbed library
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Define macros that do not conflict with STM32 HAL
#define MODEL_OK      (0)
#define MODEL_ERROR   (-1)

int8_t LIB_MODEL_Init(const void *tfliteModel, TfLiteTensor **inputTensor, uint8_t *buffer, uint32_t bufferSize);
int8_t LIB_MODEL_Run(TfLiteTensor **output);

#endif /* INC_LIB_MODEL_H_ */